"""
Additional Wrapper function for PNDM-Schedule step 
To calculate the Hallucination (https://neurips.cc/virtual/2024/poster/94558) 
Modify: diffusers/schedulers/scheduling_pndm.py 
Reference DDIMSchedule: diffusers/schedulers/scheduling_ddim.py 
"""

    def step_plms_with_pred_x0(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> dict:
        """
        A wrapper for the step_plms function that also computes the predicted x_0 value,
        and returns a dictionary containing both the previous sample and the predicted original sample.
        """
        # Call the original step_plms to obtain the prev_sample.
        # This returns a SchedulerOutput with attribute 'prev_sample'.
        step_output = self.step_plms(model_output, timestep, sample, return_dict=return_dict)
        if return_dict:
            prev_sample = step_output.prev_sample
        else:
            prev_sample = step_output[0]
        
        # For computing predicted x0, we mimic the DDIM formula.
        # Compute previous timestep index.
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        
        # Get the noise schedule values.
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        # Depending on the prediction type, compute the predicted original sample.
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"Unsupported prediction_type: {self.config.prediction_type}. "
                "It must be one of 'epsilon', 'sample', or 'v_prediction'."
            )

        # Optionally, you can perform clipping or thresholding on pred_original_sample here
        # if your configuration requires it.

        # Return both values in a dictionary.
        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}
