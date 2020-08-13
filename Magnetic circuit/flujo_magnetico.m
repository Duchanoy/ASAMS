function [B_mean] = flujo_magnetico(parameters)
[altura_dim, ancho_dim, base_dim, profundidad, no_vueltas, corriente,diam_alamb_bobina]=matsplit(parameters); 

model = mphopen('nucleo_magnetico_2_core_V2');

%mphgeom(model, 'geom1', 'view', 'auto')
cad = model.geom('geom1').feature('cad1');
cad.updateCadParamTable(true, true);

param1 = model.param.set('LL_ALTURA', altura_dim);
param2 = model.param.set('LL_ANCHO', ancho_dim);
param3 = model.param.set('LL_BASE', base_dim);
param4 = model.param.set('LL_PROFUNDIDAD', profundidad);
param5 = model.component("comp1").physics("mf").feature("coil1").set("N", no_vueltas/2);
param6 = model.component("comp1").physics("mf").feature("coil2").set("N", no_vueltas/2);
param7 = model.component("comp1").physics("mf").feature("coil1").set("ICoil", corriente);
param8 = model.component("comp1").physics("mf").feature("coil2").set("ICoil", corriente);
param9 =model.component("comp1").physics("mf").feature("coil1").set("coilWindArea", diam_alamb_bobina);
param10 =model.component("comp1").physics("mf").feature("coil2").set("coilWindArea", diam_alamb_bobina);

mesh1_size = model.mesh('mesh1').feature('size');
mesh1_size.set('custom', 'off');
mesh1_size.set('hauto', '8');

model.sol('sol1').run;

B_mean = 1e4 * mphmean(model, 'mf.normB', 'volume', 'selection',6);



