import torch
import random

"""
This is only a wrapper, not a model.
"""
class InterventionableEncoder():
    def __init__(self, model):
        self.activation = {}
        self.model = model
        self.is_cuda = next(self.model.parameters()).is_cuda

    def train(self):
        return self.model.train()
    
    def eval(self):
        return self.model.eval()
        
    def forward(
        self, 
        source, base, 
        source_mask, base_mask, 
        coords:list
    ):
        # source out + activation
        source_input_ids, source_segment_ids, source_input_mask = source
        source_outputs, intervention_activations = self.model(
            input_ids=source_input_ids, 
            token_type_ids=source_segment_ids, 
            attention_mask=source_input_mask,
            source_intervention_mask=source_mask, 
            base_intervention_mask=None, # this is a getter
            intervention_activations=None,
            intervention_coords=coords,
        )

        # base out
        base_input_ids, base_segment_ids, base_input_mask = base
        base_outputs, _ = self.model(
            input_ids=base_input_ids, 
            token_type_ids=base_segment_ids, 
            attention_mask=base_input_mask,
            source_intervention_mask=None,
            base_intervention_mask=None, 
            intervention_activations=None,
            intervention_coords=None,
        )
        
        # source -> base intervention
        counterfactual_outputs, _ = self.model(
            input_ids=base_input_ids, 
            token_type_ids=base_segment_ids, 
            attention_mask=base_input_mask,
            source_intervention_mask=source_mask,
            base_intervention_mask=base_mask, 
            intervention_activations=intervention_activations,
            intervention_coords=coords,
        )

        return source_outputs, base_outputs, counterfactual_outputs
    
"""
The following implementation is not working for multi-GPU training.
Thus, they are commented out.

Feel free to debug the following code and make it work, as it is
probably a cleaner version to do interchange intervention.

class ParallelIdPool:
    def __init__(self, max_job_id=100):
        self.job_ids = set([i for i in range(0, max_job_id)])
    
    # atomic ops
    def get(self):
        return self.job_ids.pop()
    
    # atomic ops
    def free(self, _id):
        self.job_ids.add(_id)
    
    def usable(self):
        return len(self.job_ids)

class InterventionableEncoderNotUsable(torch.nn.Module):
    def __init__(self, model, max_job_id=100):
        super().__init__()
        self.activation = {}
        self.model = model
        self.is_cuda = next(self.model.parameters()).is_cuda
        self.id_pool = ParallelIdPool(max_job_id=max_job_id)
    
    # these functions are model dependent
    # they specify how the coordinate system works
    def _coordinate_to_getter(self, mask, coords):
        handlers = []
        parallel_id = self.id_pool.get()
        for coord in coords:
            def hook(model, input, output):
                device_id = str(output.device) # map to device-based records
                self.activation[f"{device_id}-{parallel_id}-{coord}"] = output[mask]
            if self.is_cuda:
                handler = self.model.module.bert.encoder.layer[coord].output.register_forward_hook(hook)
            else:
                handler = self.model.bert.encoder.layer[coord].output.register_forward_hook(hook)
            handlers += [handler]
        return handlers, parallel_id

    def _coordinate_to_setter(self, mask, coords, parallel_id):
        handlers = []
        for coord in coords:
            def hook(model, input, output):
                device_id = str(output.device) # map to device-based records
                output[mask] = self.activation[f'{device_id}-{parallel_id}-{coord}']
            if self.is_cuda:
                handler = self.model.module.bert.encoder.layer[coord].output.register_forward_hook(hook)
            else:
                handler = self.model.bert.encoder.layer[coord].output.register_forward_hook(hook)
            handlers += [handler]
        return handlers

    def forward(
        self, 
        source, base, 
        source_mask, base_mask, 
        coords:list
    ):
        # DEBUG helpers
        # print("===")
        # print(source[0].device)
        # print(base[0].device)
        # print(source_mask.device)
        # print(base_mask.device)
        # print(torch.equal(source_mask.sum(dim=-1), base_mask.sum(dim=-1)))
        # print(coords)
        # print(self.id_pool.usable())
        # print("===")
        
        # NOTE: other ways that do not require constantly adding / removing hooks should exist

        # set hook to get activation
        # get_handler = self.names_to_layers[layer_name].register_forward_hook(self._get_activation(layer_name))
        get_handlers, parallel_id = self._coordinate_to_getter(source_mask, coords)

        # get output on source examples (and also capture the activations)
        source_outputs = self.model(*source)

        # remove the handler (don't store activations on base)
        for get_handler in get_handlers:
            get_handler.remove()

        # get base logits
        base_outputs = self.model(*base)
        
        # set hook to do the intervention
        set_handlers = self._coordinate_to_setter(base_mask, coords, parallel_id)

        # get counterfactual output on base examples
        counterfactual_outputs = self.model(*base)

        # remove the handler
        for set_handler in set_handlers:
            set_handler.remove()
            
        # cleanup
        self.id_pool.free(parallel_id)

        return source_outputs, base_outputs, counterfactual_outputs
"""