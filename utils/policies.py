import numpy as np

def linear_decay(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)

def step_decay(epoch_num, decay, decay_step, start_value):
    return start_value - (epoch_num/decay_step)*decay

def linear_growth(epoch_num, growth_start, total_epochs, start_value, end_value):
    if epoch_num < growth_start:
        return start_value
    return float(epoch_num-growth_start)/float(total_epochs-growth_start)*float(end_value-start_value)+start_value

def logistic_growth(epoch_num, start_value, k, growth_start):
    if epoch_num < growth_start:
        return start_value
    return float(1./1.+np.exp(-k*(epoch_num-growth_start)))


def custom_growth(epoch_num, growth_start, start_value, end_value):
    if epoch_num < growth_start:
        return start_value
    else:
        return end_value

