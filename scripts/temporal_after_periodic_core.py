"""Native-coefficient temporal diagnostics; no training or causal field claims."""
import numpy as np
from scripts.field_derivative_diagnostics import coefficients,weights,pde_terms,derivative


def score_coefficients(estimate,truth,frequency_weights):
    w=np.asarray(frequency_weights);true=np.sum(abs(truth)**2*w,axis=-1)
    error=np.sum(abs(estimate-truth)**2*w,axis=-1)
    floor=max(1e-24,1e-6*float(true.mean()));den=np.maximum(true,floor)
    relative=np.sqrt(error/den)
    return dict(mean=float(relative.mean()),by_time=relative.mean(0).tolist(),
                per_sample=relative.mean(1).tolist(),per_sample_time=relative.tolist(),
                absolute_rms=float(np.sqrt(error.mean())),true_rms=float(np.sqrt(true.mean())),
                normalized_squared_mean=float((error/den).mean()),floor_energy=floor,
                floor_active_fraction=float((true<floor).mean()))


def temporal_score(estimate,truth,band=None):
    if band is not None:
        return score_coefficients(coefficients(estimate)[...,:band+1],
                                  coefficients(truth)[...,:band+1],np.r_[1.,np.full(band,2.)])
    if estimate.shape!=truth.shape:raise ValueError('full-field grids differ')
    return score_coefficients(coefficients(estimate),coefficients(truth),weights(truth.shape[-1]))


def rhs_coefficients(field,c,band):
    _,diff,nonlinear=pde_terms(field,np.zeros_like(field),c,band)
    return -diff-nonlinear


def native_reference(ref,c):
    fine=ref['fields'][1024];small,large=(.0005,.001) if c['pde']=='allen_cahn' else (.01,.02)
    fd=ref['ut'][(1024,small)];wide=ref['ut'][(1024,large)]
    rhs=c['diffusion']*derivative(fine,2,c['length'])
    if c['pde']=='allen_cahn':rhs+=c['reaction']*fine*(1-fine*fine)
    else:rhs-=derivative(.5*fine*fine,1,c['length'])
    # Allen primary uses well-resolved semidiscrete RHS; Burgers uses uncertain FD.
    primary=rhs if c['pde']=='allen_cahn' else fd
    bands=dict(full=None,resolved16=16)
    uncertainty={k:temporal_score(wide,fd,b) for k,b in bands.items()}
    fd_vs_rhs={k:temporal_score(fd,rhs,b) for k,b in bands.items()}
    return primary,dict(primary='native semidiscrete RHS' if c['pde']=='allen_cahn' else 'native central FD h=.01 (uncertain)',
                        width_sensitivity=uncertainty,fd_vs_rhs=fd_vs_rhs)


def decomposition(pred,ut,ref,c,band=16):
    fine=ref['fields'][1024];truth,_=native_reference(ref,c)
    # Project native coefficients BEFORE subsampling; all modes strictly below Nyquist.
    a=coefficients(ut)[...,:band+1];b=coefficients(truth)[...,:band+1]
    rp=rhs_coefficients(pred,c,band);rt=rhs_coefficients(fine,c,band)
    et=a-b;er=rp-rt;native=b-rt;residual=a-rp
    np.testing.assert_allclose(residual,native+et-er,rtol=1e-10,atol=1e-10)
    w=np.r_[1.,np.full(band,2.)];den=np.maximum(np.sum(abs(b)**2*w,axis=-1),max(1e-24,1e-6*float(np.mean(np.sum(abs(b)**2*w,axis=-1)))))
    energy=lambda v:np.sum(abs(v)**2*w,axis=-1)
    te=energy(et);se=energy(er);cross=-2*np.sum((et*er.conj()).real*w,axis=-1)
    np.testing.assert_allclose(energy(et-er),te+se+cross,rtol=1e-10,atol=1e-10)
    return dict(time_error=score_coefficients(a,b,w),rhs_field_error=score_coefficients(rp,rt,w),
                time_error_common_normalized_squared=float((te/den).mean()),
                rhs_error_common_normalized_squared=float((se/den).mean()),
                cross_common_normalized=float((cross/den).mean()),
                discrepancy_common_normalized_squared=float((energy(et-er)/den).mean()),
                predicted_residual_rms=float(np.sqrt(energy(residual).mean())),
                native_residual_rms=float(np.sqrt(energy(native).mean())),
                oracle_time_residual_rms=float(np.sqrt(energy(b-rp).mean())),
                oracle_field_rhs_residual_rms=float(np.sqrt(energy(a-rt).mean())),
                identity_closed=True)


def analyze(pred,ut,ref,c):
    native,stability=native_reference(ref,c)
    full=temporal_score(ut,native[...,::4])
    resolved=score_coefficients(coefficients(ut)[...,:17],coefficients(native)[...,:17],np.r_[1.,np.full(16,2.)])
    secondary=score_coefficients(coefficients(ut)[...,:17],coefficients(ref['ut'][(1024,.0005 if c['pde']=='allen_cahn' else .01)])[...,:17],np.r_[1.,np.full(16,2.)])
    e=ut-native[...,::4]
    edge=np.concatenate([e[...,:4],e[...,-4:]],axis=-1)
    return dict(truth_full_sampled=full,truth_resolved_native=resolved,secondary_resolved_fd=secondary,
                temporal_edge8_energy_fraction=float(np.sum(edge*edge)/max(float(np.sum(e*e)),1e-30)),
                reference=stability,decomposition_resolved16=decomposition(pred,ut,ref,c))


def aggregate(rows):
    groups={}
    for name,r in rows.items():
        group=name.rsplit('_s',1)[0];groups.setdefault(group,[]).append(r)
    report={}
    for name,rs in sorted(groups.items()):
        report[name]=dict(count=len(rs),field32=float(np.mean([r['field32']['mean'] for r in rs])),
            time_full=float(np.mean([r['temporal']['truth_full_sampled']['mean'] for r in rs])),
            time_resolved=float(np.mean([r['temporal']['truth_resolved_native']['mean'] for r in rs])),
            time_resolved_per_seed=[r['temporal']['truth_resolved_native']['mean'] for r in rs],
            time_resolved_by_time=np.mean([r['temporal']['truth_resolved_native']['by_time'] for r in rs],axis=0).tolist(),
            own_fd_max=max(v['mean'] for r in rs for v in r['own_fd'].values()),
            time_error_energy=float(np.mean([r['temporal']['decomposition_resolved16']['time_error_common_normalized_squared'] for r in rs])),
            rhs_error_energy=float(np.mean([r['temporal']['decomposition_resolved16']['rhs_error_common_normalized_squared'] for r in rs])),
            residual_rms=float(np.mean([r['temporal']['decomposition_resolved16']['predicted_residual_rms'] for r in rs])),
            oracle_time_residual_rms=float(np.mean([r['temporal']['decomposition_resolved16']['oracle_time_residual_rms'] for r in rs])),
            oracle_field_rhs_residual_rms=float(np.mean([r['temporal']['decomposition_resolved16']['oracle_field_rhs_residual_rms'] for r in rs])))
    checks=[]
    for name,r in sorted(rows.items()):
        if not name.startswith('allen_cahn/periodic/none_s'):continue
        t=r['temporal'];d=t['decomposition_resolved16'];uncertainty=max(t['reference']['width_sensitivity']['resolved16']['mean'],t['reference']['fd_vs_rhs']['resolved16']['mean'])
        checks.append(dict(run=name,nontrivial=t['truth_resolved_native']['mean']>=.1,
            exceeds_reference_uncertainty=t['truth_resolved_native']['mean']>=10*uncertainty,
            exceeds_rhs_field_error=d['time_error_common_normalized_squared']>d['rhs_error_common_normalized_squared'],
            oracle_time_reduces_residual=d['oracle_time_residual_rms']<d['predicted_residual_rms']))
    decision=dict(checks=checks,all_three_support=len(checks)==3 and all(all(v for k,v in r.items() if k!='run') for r in checks),
                  meaning='Descriptive eligibility for a future time-constraint comparison; not causal field-error attribution, significance or verification.')
    return report,decision
