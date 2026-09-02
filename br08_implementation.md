# Numerical Mixing Tracers — Implementation & Verification
Custom E3SM branch:
```
git clone -b sorrm_w_dvd git@github.com:dylanschlichting/E3SM.git
```
## Purpose
This document explains how the `numericalMixingTracers` group is implemented, which explicitly disable non-advection tendencies. The tracers are used to quantify numerical mixing following Burchard & Rennau (2008):
  - `Mnum = (A{c^2} - (A{c})^2) / dt`
  - Implemented in `E3SM/components/mpas-ocean/src/analysis_members/mpas_ocn_discrete_variance_decay.F` where the code computes `chiSpurTracerBR08`.

## Design
- The `numericalMixingTracers` array stores 4 fields (order: `T`, `S`, `T^2`, `S^2`).
- The group is a passive, derived copy of the active tracers: it is overwritten from the active tracer state each timestep inside ```split_explicit_ab2```, then advected like any tracer.
- No explicit non-advection source/sink terms (restoring, decay, interior forcing, KPP local/non-local, frazil, runoff/sea-ice forcing) are/should be applied to the `numericalMixingTracers` group.

The passive group is populated from the active tracer state at every timestep here:
```
E3SM/components/mpas-framework/src/core_ocean/mode_forward/mpas_ocn_time_integration_split_ab2.F
  - numericalMixingTracers(1) = activeTracersNew(indexTemperature)
  - numericalMixingTracers(2) = activeTracersNew(indexSalinity)
  - numericalMixingTracers(3) = activeTracersNew(indexTemperature)**2
  - numericalMixingTracers(4) = activeTracersNew(indexSalinity)**2
```
The tracers have to be initialized at each DT to reflect the instantaneous active tracers that should then only be advected. This is how ROMS does too -- directly enforcing the values inside the master do loop. 

So, we added one new tracer group and one new analysis member. 

## Where non-advection terms are turned off
The main tracer tendency routine contains explicit checks that ensure non-advection tendencies are not applied to `numericalMixingTracers` in ```E3SM/components/mpas-ocean/src/shared/mpas_ocn_tendency.F```

- QC / forced-flag checks and explicit rejections:
  - Srface restoring, interior restoring, exponential decay, ideal age, and TTD for `numericalMixingTracers`.
- Horizontal mixing (hmix) is skipped for the group:
  - Guard: `if (trim(groupItr % memberName) /= 'numericalMixingTracers') then` before `ocn_tracer_hmix_tend`.
- KPP non-local fluxes are not applied:
  - Guarded by `if (config_use_cvmix_kpp .and. trim(groupItr % memberName) /= 'numericalMixingTracers') then` before calling the non-local flux routine.
  - Non-local flux routine: `ocn_tracer_nonlocalflux_tend`
- Implicit vertical mixing (KPP local/implicit vmix) is skipped:
  - Guard in `vmix` dispatch: `if (trim(groupItr % memberName) /= 'numericalMixingTracers') then` before calling `ocn_tracer_vmix_tend_implicit(...)`.
- Frazil and other tracer source paths skip the group similarly

## Runoff / sea-ice and the important caveat
- Bulk forcing (including river runoff, sea-ice freshwater/salinity fluxes) is computed only for `activeTracers` and `freshwaterTracers` in:
  - `E3SM/components/mpas-ocean/src/shared/mpas_ocn_surface_bulk_forcing.F`
  - The dispatch logic does **not** contain a `numericalMixingTracers` branch; therefore runoff/sea-ice fluxes are not applied directly to the passive group.

- Caveat: Although runoff/sea-ice do not directly modify `numericalMixingTracers`, the passive group is overwritten from the *active* tracer fields each timestep. 
