#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import re
from functools import partial

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Var,
    NonNegativeReals,
    NegativeReals,
    value,
)

from idaes.core import declare_process_block_class, FlowDirection, useDefault
from idaes.core.util import scaling as iscale
from idaes.core.util.exceptions import ConfigurationError
from watertap.core import (
    MembraneChannel1DBlock,
    PressureChangeType,
)
from watertap.core.membrane_channel1d import CONFIG_Template
from watertap.unit_models.unified_reverse_osmosis_base import (
    ROFlowPattern,
    UnifiedReverseOsmosisBaseData,
    _add_base_config_entries,
)
import idaes.logger as idaeslog

__author__ = "Adam Atia, Chenyu Wang"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("UnifiedReverseOsmosis1D")
class UnifiedReverseOsmosis1DData(UnifiedReverseOsmosisBaseData):
    """
    Standard 1D OARO Unit Model Class:
    - one dimensional model
    - steady state only
    - single liquid phase only
    """

    CONFIG = CONFIG_Template()

    UnifiedReverseOsmosisBaseData._add_base_config_entries(CONFIG)

    def _process_config(self):
        super()._process_config()
        if self.config.transformation_method is useDefault:
            _log.warning(
                "Discretization method was "
                "not specified for the "
                "reverse osmosis module. "
                "Defaulting to finite "
                "difference method."
            )
            self.config.transformation_method = "dae.finite_difference"

        if self.config.transformation_scheme is useDefault:
            _log.warning(
                "Discretization scheme was "
                "not specified for the "
                "reverse osmosis module."
                "Defaulting to backward finite "
                "difference."
            )
            self.config.transformation_scheme = "BACKWARD"

    def _add_membrane_channels_and_geometry(self):

        # Build membrane channel control volume
        channel_kwargs = {
            "dynamic": self.config.dynamic,
            "has_holdup": self.config.has_holdup,
            "area_definition": self.config.area_definition,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args,
            "transformation_method": self.config.transformation_method,
            "transformation_scheme": self.config.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points,
        }

        self.feed_side = MembraneChannel1DBlock(**channel_kwargs)
        self.permeate_side = MembraneChannel1DBlock(**channel_kwargs)

        self._add_length_and_width()
        add_geometry_kwargs = {
            "length_var": self.length,
            "width_var": self.width,
        }
        self.feed_side.add_geometry(
            flow_direction=FlowDirection.forward, **add_geometry_kwargs
        )
        if self.config.flow_pattern == ROFlowPattern.cocurrent:
            permeate_flow_direction = FlowDirection.backward
        elif self.config.flow_pattern == ROFlowPattern.cocurrent:
            permeate_flow_direction = FlowDirection.forward
        else:
            raise ConfigurationError(
                "The RO model supports cocurrent and countercurrent flow "
                f"patterns, but received {self.config.flow_pattern} instead."
            )
        self.permeate_side.add_geometry(
            flow_direction=permeate_flow_direction, **add_geometry_kwargs
        )
        self._add_area(include_constraint=True)

    def _add_deltaP(self, side):
        if not isinstance(side, str):
            raise TypeError(
                f"{side} is not a string. Please provide a string for the side argument."
            )

        units_meta = self.config.property_package.get_metadata().get_derived_units
        mem_side = getattr(self, side)
        setattr(
            mem_side,
            "deltaP_stage",
            Var(
                self.flowsheet().config.time,
                initialize=-1e5,
                bounds=(-1e6, 0),
                domain=NegativeReals,
                units=units_meta("pressure"),
                doc=f"Pressure drop across {side}",
            ),
        )
        if self.config.pressure_change_type == PressureChangeType.fixed_per_stage:

            @mem_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Fixed pressure drop across unit",
            )
            def eq_pressure_drop(b, t, x):
                return mem_side.deltaP_stage[t] == b.length * mem_side.dP_dx[t, x]

        else:
            # TODO why aren't we using PyomoDAE discretization to calculate this?
            @mem_side.Constraint(
                self.flowsheet().config.time, doc="Pressure drop across unit"
            )
            def eq_pressure_drop(b, t):
                return mem_side.deltaP_stage[t] == sum(
                    mem_side.dP_dx[t, x] * b.length / b.nfe
                    for x in b.difference_elements
                )

    def _add_mass_transfer(self):

        units_meta = self.config.property_package.get_metadata().get_derived_units

        # mass transfer
        def mass_transfer_phase_comp_initialize(b, t, x, p, j):
            return value(
                self.feed_side.properties[t, x].get_material_flow_terms("Liq", j)
                * self.recovery_mass_phase_comp[t, "Liq", j]
            )

        self.mass_transfer_phase_comp = Var(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=mass_transfer_phase_comp_initialize,
            bounds=(0.0, 1e6),
            domain=NonNegativeReals,
            units=units_meta("mass")
            * units_meta("time") ** -1
            * units_meta("length") ** -1,
            doc="Mass transfer to permeate",
        )

        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term",
        )
        def eq_mass_transfer_term(b, t, x, p, j):
            return (
                b.mass_transfer_phase_comp[t, x, p, j]
                == -b.feed_side.mass_transfer_term[t, x, p, j]
            )

        # Feed and permeate-side connection
        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer from feed to permeate",
        )
        def eq_connect_mass_transfer(b, t, x, p, j):
            return (
                b.permeate_side.mass_transfer_term[t, x, p, j]
                == -b.feed_side.mass_transfer_term[t, x, p, j]
            )

        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Permeate production",
        )
        def eq_permeate_production(b, t, x, p, j):
            return (
                -b.feed_side.mass_transfer_term[t, x, p, j] * b.length
                == b.flux_mass_phase_comp[t, x, p, j] * b.area
            )

    def _setup_permeate_boundary_condition(self):
        if self.permeate_side._flow_direction is FlowDirection.forward:
            x0 = self.permeate_side.length_domain.first()
        else:
            x0 = self.permeate_side.length_domain.last()

        def rule_no_flux_boundary_condition(b, t, p, j, x0):
            if (p, j) not in b.config.property_package.phase_component_set:
                return Constraint.Skip

            units_meta = b.config.property_package.get_metadata()
            flow_mol_units = units_meta.get_derived_units("flow_mol")

            props_in = b.permeate_side.properties[t, x0]
            return props_in.get_material_flow_terms(p, j) == 0 * flow_mol_units

        self.eq_no_flux_boundary_condition = Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            rule=partial(rule_no_flux_boundary_condition, x0=x0),
        )
        if not self.config.dynamic:
            state_var_type = None
            # Need a time index to define state variables
            t0 = self.flowsheet().time.first()

            bad_state_vars = ConfigurationError(
                "Encountered an unknown set of state variables. "
                "The RO model supports state variables using either "
                "component flows or component concentrations (whic )"
            )

            # TODO

            def rule_mass_fraction_flux_fraction(b, t, p, j, x0):
                if (p, j) not in b.config.property_package.phase_component_set:
                    return Constraint.Skip
                props_in = b.permeate_side.properties[t, x0]
                return (
                    b.flux_mass_phase_comp[p, j]
                    == props_in.mass_frac_phase_comp[p, j] * b.flux_mass_phase[p]
                )

    def calculate_scaling_factors(self):

        super().calculate_scaling_factors()

        for (t, x, p, j), v in self.mass_transfer_phase_comp.items():
            sf = iscale.get_scaling_factor(
                self.feed_side.properties[t, x].get_material_flow_terms(p, j)
            )
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)
            v = self.feed_side.mass_transfer_term[t, x, p, j]
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)
            v = self.permeate_side.mass_transfer_term[t, x, p, j]
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)

        if hasattr(self.feed_side, "deltaP_stage"):
            for v in self.feed_side.deltaP_stage.values():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-4)
        if hasattr(self.permeate_side, "deltaP_stage"):
            for v in self.permeate_side.deltaP_stage.values():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-4)
        if self.config.pressure_change_type == PressureChangeType.fixed_per_stage:
            # TODO additional scaling
            pass
        elif hasattr(self, "eq_pressure_drop"):
            for t, condata in self.eq_pressure_drop.items():
                sf_dP = iscale.get_scaling_factor(self.deltaP[t])
                iscale.constraint_scaling_transform(condata, sf_dP)

        for (t, x, p, j), condata in self.eq_permeate_production.items():
            sf = iscale.get_scaling_factor(
                self.permeate_side.properties[t, x].get_material_flow_terms(p, j),
                default=1,
            )
            iscale.constraint_scaling_transform(condata, sf)

        for (t, x, p, j), condata in self.eq_mass_transfer_term.items():
            sf = min(
                iscale.get_scaling_factor(
                    self.mass_transfer_phase_comp[t, x, p, j], default=1
                ),
                iscale.get_scaling_factor(
                    self.feed_side.mass_transfer_term[t, x, p, j], default=1
                ),
            )
            iscale.constraint_scaling_transform(condata, sf)
        for (t, x, p, j), condata in self.eq_connect_mass_transfer.items():
            sf = iscale.get_scaling_factor(
                self.feed_side.mass_transfer_term[t, x, p, j], default=1
            )
            iscale.constraint_scaling_transform(condata, sf)
