"""Numerical safeguards for the compact radar experiment."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from alta_radar_models import (RadarConfig,make_model,warp_rate,encode_inputs,
    weighted_loss,save_bundle,load_bundle,predict_ensemble,architecture_manifest,
    physical_mean_mse)


def test_missing_radar_is_explicit_and_excluded_from_loss():
    x=np.zeros((1,9,32,32),np.float32);x[0,0,0,0]=np.nan
    a=encode_inputs(x)
    assert a[0,0,1,0,0]==0
    assert a[0,0,1,0,1]==1
    q=np.ones_like(x)*.4
    aq=encode_inputs(np.stack([x,q],2))
    assert np.isclose(float(aq[0,0,1,0,1]),.4)
    assert aq[0,0,1,0,0]==0
    target=torch.tensor([1.,999.]);mask=torch.tensor([1.,0.])
    assert float(weighted_loss(torch.tensor([1.,-99.]),target,mask))==0.


def test_advection_moves_the_feature_with_correct_sign():
    source=torch.zeros((1,1,9,9));source[0,0,4,4]=2.
    flow=torch.zeros((1,2,9,9));flow[:,0]=1.
    shifted=warp_rate(source,flow)
    assert torch.allclose(shifted[0,0,4,5],torch.tensor(2.))
    assert shifted[0,0,4,4].abs()<1e-6
    assert torch.allclose(warp_rate(source,torch.zeros_like(flow)),source,atol=1e-6)


def test_all_families_have_finite_multilead_outputs_and_gradients():
    torch.set_num_threads(2)
    x=encode_inputs(np.random.default_rng(1).exponential(.3,(2,9,32,32)).astype(np.float32))
    for name in ("convlstm","dgmr","nowcastnet"):
        model=make_model(RadarConfig(name=name,output_frames=4,base_channels=4))
        out=model(x)
        assert out.shape==(2,4,32,32)
        assert torch.isfinite(out).all() and out.min()>=0
        out.square().mean().backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads)>=10 and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
        if name=="nowcastnet":
            # The stochastic decoder intentionally detaches evolution.
            assert all(p.grad is None for p in model.evolution.parameters())
            evo,flow=model.evolution(x)
            (evo.mean()+.01*flow.square().mean()).backward()
            assert any(p.grad is not None for p in model.evolution.parameters())


def test_checkpoint_roundtrip_reproduces_seeded_ensemble(tmp_path):
    torch.set_num_threads(2)
    cfg=RadarConfig(name="dgmr",output_frames=3,base_channels=4)
    model=make_model(cfg)
    # Spectral-normalization vectors initialize on a training forward pass.
    model(encode_inputs(np.ones((1,9,32,32),np.float32)*.1))
    model.eval()
    bundle={"model":model,"config":cfg,"manifest":architecture_manifest(cfg,model)}
    x=np.ones((1,9,32,32),np.float32)*.1
    path=tmp_path/"model.pt"
    save_bundle(bundle,path)
    before=predict_ensemble(bundle,x,members=2,seed=12)
    after=predict_ensemble(load_bundle(path),x,members=2,seed=12)
    np.testing.assert_array_equal(before,after)
    assert np.isfinite(after).all()
    assert not np.array_equal(after[:,0],after[:,1])


def test_direct_physical_head_retains_gradient_at_large_decoder_output():
    torch.set_num_threads(2)
    cfg=RadarConfig(name="dgmr",output_frames=2,base_channels=4,linear_rate_head=True)
    model=make_model(cfg)
    torch.nn.init.zeros_(model.output.weight)
    torch.nn.init.constant_(model.output.bias,10.)
    out=model(encode_inputs(np.ones((1,9,32,32),np.float32)*.1))
    rate=torch.expm1(out)
    assert torch.allclose(rate,torch.full_like(rate,10.),atol=.001)
    loss=physical_mean_mse([out],torch.zeros_like(out),torch.ones_like(out))
    loss.backward()
    assert torch.isfinite(model.output.bias.grad).all()
    assert model.output.bias.grad.abs().sum()>1.


if __name__=="__main__":
    import tempfile
    test_missing_radar_is_explicit_and_excluded_from_loss()
    test_advection_moves_the_feature_with_correct_sign()
    test_all_families_have_finite_multilead_outputs_and_gradients()
    test_direct_physical_head_retains_gradient_at_large_decoder_output()
    with tempfile.TemporaryDirectory() as directory:
        test_checkpoint_roundtrip_reproduces_seeded_ensemble(Path(directory))
    print("Five Alta radar numerical checks passed.")
