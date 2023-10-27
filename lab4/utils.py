import numpy as np

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    ... (unchanged code)

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Wolfe':
            alpha = self.wolfe_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Armijo':
            alpha = self.armijo_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Constant':
            alpha = self.constant_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Best':
            alpha = self.best_line_search(oracle, x_k, d_k, previous_alpha)
        else:
            raise ValueError('Unknown method {}'.format(self._method))
        
        return alpha

    def wolfe_line_search(self, oracle, x_k, d_k, previous_alpha):
        c1, c2, alpha_0 = self.c1, self.c2, self.alpha_0
        alpha = alpha_0

        while True:
            phi_alpha = oracle.func_directional(x_k + alpha * d_k, d_k)
            phi_0 = oracle.func(x_k)

            if (phi_alpha > phi_0 + c1 * alpha * oracle.grad_directional(x_k, d_k).dot(d_k) or
                    (phi_alpha >= phi_0 and alpha > 0)):
                return self.zoom(oracle, x_k, d_k, alpha, previous_alpha, c1, c2)
            
            grad_phi_alpha = oracle.grad_directional(x_k + alpha * d_k, d_k).dot(d_k)
            
            if abs(grad_phi_alpha) <= -c2 * oracle.grad_directional(x_k, d_k).dot(d_k):
                return alpha
            
            if grad_phi_alpha >= 0:
                return self.zoom(oracle, x_k, d_k, alpha, previous_alpha, c1, c2)

            previous_alpha = alpha
            alpha = 2 * alpha

    def armijo_line_search(self, oracle, x_k, d_k, previous_alpha):
        c1, alpha_0 = self.c1, self.alpha_0
        alpha = alpha_0

        while True:
            phi_alpha = oracle.func_directional(x_k + alpha * d_k, d_k)
            phi_0 = oracle.func(x_k)
            
            if phi_alpha <= phi_0 + c1 * alpha * oracle.grad_directional(x_k, d_k).dot(d_k):
                return alpha
            
            if alpha < 1e-12:
                return None  # Step size is too small

            alpha /= 2.0

    def constant_line_search(self, oracle, x_k, d_k, previous_alpha):
        return self.c

    def best_line_search(self, oracle, x_k, d_k, previous_alpha):
        # Use oracle.minimize_directional() for optimal step size
        result = oracle.minimize_directional(x_k, d_k)
        if result is not None:
            return result[0]  # The first element is the optimal step size
        else:
            return None

    def zoom(self, oracle, x_k, d_k, alpha_high, alpha_low, c1, c2):
        while True:
            alpha = 0.5 * (alpha_low + alpha_high)
            phi_alpha = oracle.func_directional(x_k + alpha * d_k, d_k)
            phi_0 = oracle.func(x_k)
            grad_phi_alpha = oracle.grad_directional(x_k + alpha * d_k, d_k).dot(d_k)

            if (phi_alpha > phi_0 + c1 * alpha * oracle.grad_directional(x_k, d_k).dot(d_k) or
                    phi_alpha >= phi_alpha(alpha_low)):
                alpha_high = alpha
            else:
                if abs(grad_phi_alpha) <= -c2 * oracle.grad_directional(x_k, d_k).dot(d_k):
                    return alpha
                if grad_phi_alpha * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha
