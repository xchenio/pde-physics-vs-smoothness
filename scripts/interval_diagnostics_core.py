"""Native resolved interval diagnostics, no losses/training/model selection."""
import numpy as np
from scripts.field_derivative_diagnostics import coefficients,weights
from scripts.temporal_after_periodic_core import score_coefficients,rhs_coefficients


def simpson(values,times):
    times=np.asarray(times,dtype=float);n=len(times)
    if n<3 or n%2!=1:raise ValueError('odd Simpson count >=3 required')
    steps=np.diff(times)
    if np.any(steps<=0) or not np.allclose(steps,steps[0],rtol=1e-8,atol=1e-12):raise ValueError('uniform increasing nodes required')
    w=np.ones(n);w[1:-1:2]=4;w[2:-1:2]=2
    return np.sum(values*w.reshape((1,)*(values.ndim-2)+(n,1)),axis=-2)*steps[0]/3


def indices(times,requested):
    ts=np.asarray(times);ix=np.array([np.argmin(abs(ts-t)) for t in requested],dtype=int)
    if np.max(abs(ts[ix]-requested))>1e-10:raise ValueError('reference time absent')
    return ix


def interval_metrics(pred,model_times,ref,c,window,band=16):
    mt=np.asarray(window['model_times']);rt=np.asarray(window['reference_times'])
    v=pred[:,indices(model_times,mt)];truth=ref['fields'][1024][:,indices(ref['times'],rt)]
    w=np.r_[1.,np.full(band,2.)]
    dp=coefficients(v[:,-1]-v[:,0])[...,:band+1]
    dt=coefficients(truth[:,-1]-truth[:,0])[...,:band+1]
    ip=simpson(rhs_coefficients(v,c,band),mt);it=simpson(rhs_coefficients(truth,c,band),rt)
    ipc=simpson(rhs_coefficients(v[:,::2],c,band),mt[::2]);itc=simpson(rhs_coefficients(truth[:,::2],c,band),rt[::2])
    # score_coefficients expects sample x time x coefficients.
    sc=lambda a,b:score_coefficients(a[:,None,:],b[:,None,:],w)
    zero=np.zeros_like(dt);et=dp-dt;er=ip-it;native=dt-it;residual=dp-ip
    np.testing.assert_allclose(residual,native+et-er,rtol=1e-10,atol=1e-10)
    energy=lambda a:np.sum(abs(a)**2*w,axis=-1)
    den=np.maximum(energy(dt),max(1e-24,1e-6*float(energy(dt).mean())))
    te=energy(et);re=energy(er);cross=-2*np.sum((et*er.conj()).real*w,axis=-1)
    np.testing.assert_allclose(energy(et-er),te+re+cross,rtol=1e-10,atol=1e-10)
    uncertainty=dict(native_integral_defect=sc(native+dt,dt),reference_quadrature=sc(it-itc+dt,dt))
    if 1024 in ref.get('coarse_fields',{}):
        coarse=ref['coarse_fields'][1024][:,indices(ref['times'],rt)]
        dc=coefficients(coarse[:,-1]-coarse[:,0])[...,:band+1]
        ic=simpson(rhs_coefficients(coarse,c,band),rt)
        uncertainty['solver_increment']=sc(dc,dt)
        uncertainty['solver_rhs_integral']=sc(ic-it+dt,dt)
    if 512 in ref['fields'] and c['pde']=='allen_cahn':
        space=ref['fields'][512][:,indices(ref['times'],rt)]
        ds=coefficients(space[:,-1]-space[:,0])[...,:band+1]
        isp=simpson(rhs_coefficients(space,c,band),rt)
        uncertainty['space_increment']=sc(ds,dt)
        uncertainty['space_rhs_integral']=sc(isp-it+dt,dt)
    uncertainty_max=max(r['mean'] for r in uncertainty.values())
    increment=sc(dp,dt);quad=sc(ip-ipc+dt,dt)
    # Fixed one-shot diagnostic: replace only low-mode endpoint increment by
    # integrated model RHS. Uses NO true future value; not an exported predictor.
    end_cf=coefficients(v[:,-1]);repair_cf=end_cf.copy();repair_cf[...,:band+1]-=residual
    repaired=np.fft.irfft(repair_cf,n=256,norm='forward')
    true_end=truth[:,-1,::4]
    field=lambda z:score_coefficients(coefficients(z)[:,None],coefficients(true_end)[:,None],weights(256))
    return dict(center=window['center'],half_width=window['half_width'],increment=increment,
                integral_rhs_error=sc(ip-it+dt,dt),model_quadrature=quad,reference_uncertainty=uncertainty,
                reference_uncertainty_max=uncertainty_max,
                time_increment_common_energy=float((te/den).mean()),
                state_rhs_integral_common_energy=float((re/den).mean()),
                cross_common_energy=float((cross/den).mean()),
                error_discrepancy_common_energy=float((energy(et-er)/den).mean()),
                predicted_defect_rms=float(np.sqrt(energy(residual).mean())),
                native_defect_rms=float(np.sqrt(energy(native).mean())),
                oracle_increment_defect_rms=float(np.sqrt(energy(dt-ip).mean())),
                oracle_state_rhs_defect_rms=float(np.sqrt(energy(dp-it).mean())),
                endpoint_field=field(v[:,-1]),fixed_integral_repair_field=field(repaired),
                descriptive_gate=bool(increment['mean']>=.1 and increment['mean']>=10*uncertainty_max and
                    quad['mean']<=.1*increment['mean'] and increment['floor_active_fraction']==0),
                coefficient_identity_closed=True)


def aggregate(rows):
    report={};groups={}
    for name,r in rows.items():groups.setdefault(name.rsplit('_s',1)[0],[]).append(r)
    for name,rs in sorted(groups.items()):
        windows=[w for r in rs for w in r['intervals']];primary=[w for w in windows if not w['secondary_late']]
        report[name]=dict(count=len(rs),midpoint_field=float(np.mean([r['midpoint_field']['field']['relative_mean'] for r in rs])),
            midpoint_time_resolved=float(np.mean([r['midpoint_time_resolved']['mean'] for r in rs])),
            late_field=float(np.mean([r['late_field']['field']['relative_mean'] for r in rs])),
            late_time_resolved=float(np.mean([r['late_time_resolved']['mean'] for r in rs])),
            interval_increment=float(np.mean([w['increment']['mean'] for w in primary])),
            interval_increment_per_seed=[float(np.mean([w['increment']['mean'] for w in r['intervals'] if not w['secondary_late']])) for r in rs],
            interval_native_uncertainty=float(np.mean([w['reference_uncertainty_max'] for w in primary])),
            interval_model_quadrature=float(np.mean([w['model_quadrature']['mean'] for w in primary])),
            increment_common_energy=float(np.mean([w['time_increment_common_energy'] for w in primary])),
            rhs_integral_common_energy=float(np.mean([w['state_rhs_integral_common_energy'] for w in primary])),
            cross_common_energy=float(np.mean([w['cross_common_energy'] for w in primary])),
            endpoint_field=float(np.mean([w['endpoint_field']['mean'] for w in primary])),
            repair_field=float(np.mean([w['fixed_integral_repair_field']['mean'] for w in primary])),
            center_h_gates=[dict(run_seed=i,center=w['center'],half_width=w['half_width'],passed=w['descriptive_gate']) for i,r in enumerate(rs) for w in r['intervals']])
    decision={}
    for name in ['allen_cahn/periodic/none','burgers/linear/low']:
        if name not in groups:continue
        rs=groups[name];primary=[w for r in rs for w in r['intervals'] if not w['secondary_late']]
        decision[name]=dict(all_primary_centers_windows_seeds=len(rs)==3 and all(w['descriptive_gate'] for w in primary),
            pure_increment_dominance_all=all(w['time_increment_common_energy']>w['state_rhs_integral_common_energy'] for w in primary),
            limits='Descriptive eligibility only. A failed strict all-center gate does not prove accurate dynamics; never choose favorable centers. No training authorized by screen alone.')
    return report,decision
