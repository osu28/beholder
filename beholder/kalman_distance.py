# https://www.kalmanfilter.net/kalman1d.html

class KalmanDistance:
    """Retains the filter and distance information for a particular track.
    """

    def __init__(self, measured_dist: float) -> None:
        """Initialize Kalman Filter params

        Args:
            measured_dist (float): the measured distance to use to initialize
            the filter.
        """
        self.x_n_n = measured_dist      # ititial distance, significantly helps
        self.p_n_n = 10**2              # initial estimate uncertainty
        self.q = 0.8                    # process uncertainty
        self.r_n = 10**2                # measurement uncertainty

        eeu = self.p_n_n + self.q       # extrapolated estimate uncertainty
        self.K = eeu / (eeu + self.r_n)  # Kalman gain

    def update(self, measured_dist: float) -> float:
        """Update the Kalman Filter and return the filtered estimate.

        Args:
            measured_dist (float): The measured prediction from the model.

        Returns:
            float: Distance estimate of the current state.
            float: Estimate uncertainty of the current state.
        """

        # reject large changes entirely
        if abs(measured_dist - self.x_n_n) > 4:
            return self.x_n_n, self.p_n_n

        # kalman gain
        self.K = self.p_n_n / (self.p_n_n + self.r_n)
        self.x_n_n = self.x_n_n + self.K * \
            (measured_dist - self.x_n_n)     # state update
        # covariance update
        self.p_n_n = (1 - self.K) * self.p_n_n

        # predict and move to next state
        self.p_n_n = self.p_n_n + self.q

        # print(f'{self.x_n_n:.2f}\t{self.p_n_n:.2f}\t{self.K:.2f}')
        return (self.x_n_n, self.p_n_n)

    def get_dist_pred(self) -> float:
        """Helper function for getting the most recent filtered predicted
        distance.

        Returns:
            float: Distance estimate of the current state.
            float: Estimate uncertainty of the current state.
        """
        return (self.x_n_n, self.p_n_n)


class KalmanDistanceFilter:
    """Stores the kalman filters for each track in a dictionary.
    """
    stored = dict()

    def update_filters(self, distance_preds) -> list:
        """Loop through distance predictions and pass them into Kalman Filters.
        Return the result.

        Args:
            distance_preds (list): List of objects of the format:
                [track_id, label, [roi], distance]

        Returns:
            list: The same list as the argument distance_preds but with updated
            distances.
        """
        for i, (track_id, label, roi, pred) in enumerate(distance_preds):
            if (track_id, label) in self.stored:
                # exists, update
                new_dist, _ = self.stored[(track_id, label)].update(pred)
                # print(f'{distance_preds[i][3]:.2f} -> {new_dist:.2f}')
                distance_preds[i][3] = new_dist
            else:
                # DNE, create
                self.stored[(track_id, label)] = KalmanDistance(pred)

        return distance_preds
