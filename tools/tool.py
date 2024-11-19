
def preprocess_teacher(model, teacher):
    for param_m, param_t in zip(model.parameters(), teacher.parameters()):
        param_t.data.copy_(param_m.data)  # initialize
        param_t.requires_grad = False  # not update by gradient


# def update_ema_variables(self, model1, model2):
#         # Use the true average until the exponential average is more correct
#         alpha = min(1 - 1 / (self.iteration + 1), self.m)
#         #print(alpha)
#         for t_param, s_param in zip(model2.decoder.parameters(), model1.decoder.parameters()):
#             t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
#         for t_param, s_param in zip(model2.aspp.parameters(), model1.aspp.parameters()):
#             t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
#         for t_param, s_param in zip(model2.backbone.parameters(), model1.backbone.parameters()):
#             t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)

def update_ema_variables(model, teacher, momentum=0.9995, global_step=2000):
    momentum = min(1 - 1 / (global_step + 1), momentum)
    for ema_param, param in zip(teacher.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(1 - momentum, param.data)