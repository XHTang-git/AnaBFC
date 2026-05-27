
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'AnaBFC':
        assert(opt.dataset_mode == 'unaligned')
        from .AnaBFC_model import AnaBFC
        model = AnaBFC()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
