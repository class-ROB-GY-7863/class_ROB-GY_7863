
import numpy as np
import matplotlib.pyplot as plt

# estimators
class Estimator:
    def estimate(self, xhat_k, P_k, u_kp1, y_kp1):
        raise NotImplementedError

class EmptyEstimator(Estimator):
    def estimate(self, xhat_k, P_k, u_kp1, y_kp1, system):
        return xhat_k, P_k

class KalmanFilter(Estimator):
    def estimate(self, xhat_k, P_k, u_kp1, y_kp1, system):
        A, B, C, Sigma_x, Sigma_y = system.A, system.B, system.C, system.Sigma_x, system.Sigma_y
        # Predict
        xhat_prior = A @ xhat_k + B @ u_kp1
        P_prior = A @ P_k @ A.T + Sigma_x
        # Innovation
        S = C @ P_prior @ C.T + Sigma_y
        K = P_prior @ C.T @ np.linalg.inv(S)
        y_pred = C @ xhat_prior
        innov = y_kp1 - y_pred
        # Update
        xhat_post = xhat_prior + K @ innov
        I = np.eye(xhat_k.shape[0])
        P_post = (I - K @ C) @ P_prior
        return xhat_post, P_post

class EnsembleKalmanFilter(Estimator):
    def estimate(self, xhat_k, P_k, u_kp1, y_kp1, system):
        raise NotImplementedError