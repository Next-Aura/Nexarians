import numpy as np
from typing import Literal

class AdvancedConfig:
    def __init__(self,
                patience: int = 5,
                lr_scheduler: Literal['constant', 'plateau', 'invscaling', 'cosine', 'onecycle'] = 'constant',
                adaptive_alpha: bool = False,
                power_t: float = 0.25,
                factor: float = 0.1,
                asymmetric_bias: bool = False,
                symmetric_bias: bool = False,
                reg_target: Literal['weight', 'bias'] | None='weight'
                ):
        
        self.lr_sche = lr_scheduler
        self.adaptive_alpha = bool(adaptive_alpha)
        self.power_t = float(power_t)
        self.factor = float(factor)
        self.asymm_b = bool(asymmetric_bias)
        self.symm_b = bool(symmetric_bias)
        self.reg_target = reg_target

        self.patience = int(patience)
        self.wait = 0
        self.tol = 1e-8
        self.iter = 0
        self.alpha = None
        self.lr_rate = None
        self.loss = []
        self.wait = 0

        self.best_loss = np.inf

    def compute(self, 
                input_iter: int=None, 
                alpha_input: None=None, 
                lr_input: None=None, 
                loss_input: None=None, 
                tol_input: None=None, 
                max_iter_input: None=None,
                b_input: None=None,
                w_input: None=None):
        self.iter = input_iter
        self.loss = loss_input
        self.lr_rate = lr_input
        self.alpha = alpha_input
        self.tol = tol_input
        self.max_iter = max_iter_input
        self.b = b_input if b_input is not None else "NOVALUE"
        self.w = w_input if w_input is not None else "NOVALUE"

        if self.asymm_b and np.any(self.b) != "NOVALUE":
            bias = self.asymm_bias()

        if self.symm_b and np.any(self.b) != "NOVALUE":
            bias = self.symm_bias()

        if self.asymm_b and self.symm_b:
            raise ValueError(f"Both symmetric_bias and asymmetric_bias cannot be activate at the same time")

        if self.asymm_b and np.any(self.w) != "NOVALUE":
            weights = self.asymm_bias()

        if self.symm_b and np.any(self.w) != "NOVALUE":
            weights = self.symm_bias()

        return {
            "lr": self.lr_scheduler() if self.lr_rate is not None and self.lr_sche else lr_input,
            "alpha": self.ada_alpha() if self.alpha is not None and self.adaptive_alpha else alpha_input,
            "b": bias if self.iter > 10 and (self.symm_b or self.asymm_b) and np.any(self.b) != "NOVALUE" and self.reg_target == 'bias' else self.b,
            "w": weights if self.iter > 10 and (self.symm_b or self.asymm_b) and np.any(self.w) != "NOVALUE" and self.reg_target == 'weight' else self.w,
        }

    def lr_scheduler(self):
        if self.lr_sche == 'constant':
            lr = self.lr_rate
        
        elif self.lr_sche == 'invscaling':
            lr = self.lr_rate / ((self.iter + 1)**self.power_t + 1e-8)
        
        elif self.lr_sche == 'plateau':
            if self.iter > 0:
                current_loss = self.loss[-1]
                if current_loss < self.best_loss - 1e-8:
                    self.best_loss = current_loss
                    self.wait = 0

                elif abs(current_loss - self.best_loss) < self.tol:
                    self.wait += 1

                else:
                    self.wait = 0

                if self.wait >= self.patience:
                    lr = self.lr_rate * self.factor
                    self.wait = 0

                else:
                    lr = self.lr_rate

            else:
                lr = self.lr_rate


        elif self.lr_sche == 'cosine':
            lr = 0.5 * self.lr_rate * (1 + np.cos(np.pi * self.iter / self.max_iter))

        elif self.lr_sche == 'onecycle':
            lr = self.lr_rate * np.cos(np.pi * self.iter / self.max_iter)    
        else:
            raise ValueError(f"Invalid lr_scheduler argument {self.lr_sche}")

        return lr

    def ada_alpha(self):
        self.alpha = self.alpha * (self.loss[-1] / (self.loss[-2] + 1e-8)) if self.adaptive_alpha else self.alpha
        return self.alpha
    
    def asymm_bias(self):
        if self.reg_target == 'bias':
            if np.mean(self.loss[-5:]) > np.mean(self.loss[-10:-5]):
                self.b += np.sign(self.b) * self.alpha

            else:
                self.b -= np.sign(self.b) * self.alpha

            return self.b

        elif self.reg_target == 'weight':
            if np.mean(self.loss[-5:]) > np.mean(self.loss[-10:-5]):
                self.w += np.sign(self.w) * self.alpha

            else:
                self.w -= np.sign(self.w) * self.alpha

            return self.w
    
        else:
          raise ValueError(f"Invalid reg_target argument {self.reg_target}")
        
    
    def symm_bias(self):
        if self.reg_target == 'bias':
            if np.mean(self.loss[-5:]) > np.mean(self.loss[-10:-5]):
                self.b -= np.sign(self.b) * self.alpha

            else:
                self.b += np.sign(self.b) * self.alpha

            return self.b

        elif self.reg_target == 'weight':
            if np.mean(self.loss[-5:]) > np.mean(self.loss[-10:-5]):
                self.w -= np.sign(self.w) * self.alpha

            else:
                self.w += np.sign(self.w) * self.alpha

            return self.w
        
        else:
            raise ValueError(f"Invalid reg_target argument {self.reg_target}")