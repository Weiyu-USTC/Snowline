"""Controlled capacity diagnostic, not a model-selection or deployment experiment.

October + lagged RONI + strong indicators predict all Nov-Apr months. Compare
training error and omitted-season error, using the exact same 20 seasonal rows.
"""
import json
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from monthly import ROOT,MONTHS,SEED,load_training,make_features


def main():
    frame=load_training()
    X,_=make_features(frame,0,True)
    Y=frame[MONTHS[1:]].to_numpy(float)
    rows=[]
    for label,hidden,alpha,steps in [('strong_regularization',(4,),20.,250),
                                    ('flexible_low_penalty',(64,64),1e-8,2000)]:
        train_squared=[];test_squared=[];warning_count=0
        with threadpool_limits(limits=1):
            for held in range(len(frame)):
                keep=np.arange(len(frame))!=held
                xmean=X[keep].mean(0);xscale=X[keep].std(0);xscale[xscale<1e-8]=1
                ymean=Y[keep].mean(0);yscale=Y[keep].std(0);yscale[yscale<1e-8]=1
                model=MLPRegressor(hidden_layer_sizes=hidden,activation='tanh',solver='lbfgs',
                    alpha=alpha,max_iter=steps,max_fun=20000,random_state=SEED,tol=1e-8)
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit((X[keep]-xmean)/xscale,(Y[keep]-ymean)/yscale)
                    warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in observed)
                train_prediction=model.predict((X[keep]-xmean)/xscale)*yscale+ymean
                test_prediction=np.maximum(0.,model.predict((X[held:held+1]-xmean)/xscale)*yscale+ymean)
                train_squared.extend(((train_prediction-Y[keep])**2).ravel())
                test_squared.extend(((test_prediction-Y[held:held+1])**2).ravel())
        rows.append({'configuration':label,'hidden_layers':list(hidden),'weight_penalty':alpha,
                     'max_iter':steps,'training_rmse_inches':float(np.sqrt(np.mean(train_squared))),
                     'held_out_rmse_inches':float(np.sqrt(np.mean(test_squared))),
                     'convergence_warnings':warning_count})
    output={'purpose':'Explain training-vs-generalization error; not selecting a winner on test performance.',
            'experiment':'20-season LOSO, October context to all November-April targets',
            'seed':SEED,'results':rows}
    (ROOT/'outputs/neural_overfit_diagnostic.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
