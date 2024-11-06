from src.models import promptmr, prompt_sme

def build_sensitivity_model(args):
    # to maximize channel within 8GB VRAM limitation
    if not args.crop_by_width:
        opt1, opt2, opt3 = 1, 2, 1
    else:
        opt1, opt2, opt3 = 1, 0, 0

    sens_model = prompt_sme.SensitivityModel(num_adj_slices=1,
                                            n_feat0 =int(4*args.sens_chans),
                                            feature_dim = [int(6*args.sens_chans)+opt1, int(7*args.sens_chans)+opt2, int(9*args.sens_chans)+opt3],
                                            prompt_dim = [int(2*args.sens_chans)+opt1, int(3*args.sens_chans)+opt2, int(5*args.sens_chans)+opt3],
                                            len_prompt = [5, 5, 5],
                                            prompt_size = [8, 4, 2],
                                            n_enc_cab = [2, 3, 3],
                                            n_dec_cab = [2, 2, 3],
                                            n_skip_cab = [1, 1, 1],
                                            n_bottleneck_cab = 3,
                                            no_use_ca = None,
                                            mask_center = True,
                                            low_mem = False)
    return sens_model

def build_recon_model(num_cascades, chans):
    model = promptmr.PromptMR(
        num_cascades = num_cascades,
        num_adj_slices = 1,
        n_feat0 = 8*chans,
        feature_dim = [12*chans, 16*chans, 20*chans],
        prompt_dim = [4*chans, 8*chans, 12*chans],
        len_prompt = [5, 5, 5],
        prompt_size = [8*chans, 4*chans, 2*chans],
        n_enc_cab = [2, 3, 3],
        n_dec_cab = [2, 2, 3],
        n_skip_cab = [1, 1, 1],
        n_bottleneck_cab = 3
    )
    
    return model
    
def build_model(args):
    sens_model = build_sensitivity_model(args)
    model1 = build_recon_model(args.cascade1, args.chans1)
    model2 = build_recon_model(args.cascade2, args.chans2)
    model3 = build_recon_model(args.cascade3, args.chans3)

    return {'sens_net': sens_model,
            'model1': model1,
            'model2': model2,
            'model3': model3}