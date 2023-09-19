##### case: dump

def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None):
    
    ################ debug starts
    import dill
    with open('function_arguments.pkl', 'wb') as f:
        # `locals()` returns the function's local symbol table
        # We will create a dictionary without `self` since we'll handle it separately
        args = {k: v for k, v in locals().items() if k != 'self'}
        dill.dump(args, f)
    
    # If you also want to save the entire `self` object:
    with open('self_object.pkl', 'wb') as f:
        dill.dump(self, f)
    
    exit()
    ################ debug ends


    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = self.model.apply_model(x, t, c)


###### case: load

# Load serialized data
with open('aaaa.pkl', 'rb') as f:
    saved_args = dill.load(f)

with open('self_object.pkl', 'rb') as f:
    saved_self = dill.load(f)

记住一定要将 function_arguments.pkl 的名字改成一个新的名字 例如aaa.pkl
可以用mv function_arguments.pkl aaa.pkl