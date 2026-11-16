"""Use exported model coefficients without scikit-learn or unsafe pickle loading."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Exploratory November-April snowfall estimate, inches.')
    parser.add_argument('--dataset', choices=['copper','dillon','climax'], default='copper')
    parser.add_argument('--model', choices=['climatology','october_linear','enso_linear','october_enso_ridge'],default='climatology')
    parser.add_argument('--october', type=float, help='Observed October snowfall, inches')
    parser.add_argument('--jas-oni', type=float, help='Published JAS ONI, °C; not winter DJF')
    parser.add_argument('--models',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/trained_models.json')
    args = parser.parse_args()
    if args.october is not None and (not math.isfinite(args.october) or args.october < 0):
        parser.error('October snowfall must be finite and nonnegative.')
    if args.jas_oni is not None and not math.isfinite(args.jas_oni):
        parser.error('JAS ONI must be finite.')
    bundle=json.loads(args.models.read_text())
    dataset=bundle['datasets'][args.dataset]
    model=dataset['models'][args.model]
    features={'oct':args.october,'oni_jas':args.jas_oni}
    if any(features[k] is None for k in model['features']):
        parser.error('Selected model requires ' + ', '.join(model['features']))
    estimate=model['intercept']+sum(c*features[k] for k,c in zip(model['features'],model['coefficients']))
    output={'dataset':args.dataset,'model':args.model,'nov_apr_inches':round(estimate,1),
            'training_seasons':model['n_train'], 'last_training_start_year':model['train_last_year'],
            'status':bundle['warning'], 'inputs':features}
    if args.model == 'climatology':
        output['prediction_interval80_inches']=[round(v,1) for v in dataset['baseline_prediction_interval80']]
        output['interval_assumption']='Student-t predictive interval assuming independent, stable, approximately normal seasonal totals; not calibrated for climate change.'
    else:
        output['interval_note']='No validated prediction interval for this experimental regression.'
    if estimate < 0:
        output['extrapolation_warning']='Negative regression prediction is physically implausible. Do not use this extrapolation.'
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
