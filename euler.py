import numpy as np


def euler(odefun, tspan, y0):
    """Uses Euler's Method to calculate the solution to an ODE.

    Parameters
    ----------
    odefun : callable
        A callable function to the derivative function defining the system.
    tspan : array_like
        An array of times (or other independent variable) at which to evaluate
        Euler's Method.  The num_times values in this array determine the size
        of the outputs.
    y0 : array_like
        Array containing the initial conditions to be used in evaluating the
        odefun.  Must be the same size as that expected for the
        second input of odefun.

    Returns
    -------
    t : ndarray
        t[i] is the ith time (or other independent variable) at which
        Euler's Method was evaluated.
    y : ndarray
        y[i] is the ith dependent variable at which Euler's Method was
        evaluated.

    Notes
    -----
    Uses Euler's Method to calculate the solution to the ODE defined by the
    system of equations odefun, over the time steps defined by `tspan`, with the
    initial conditions `y0`.  The time output is stored in the array `t`, while
    the dependent variables are stored in the array `y`.  Euler's Method uses
    the equation:

                       dY
       Y[i+1] = Y[i] + -- delta_t
                       dt

    where delta_t is the time step, Y[i] is the values of the dependent
    variables at the current time, Y[i+1] is the values of the dependent
    variables at the next time, and dY/dt is the derivative function evaluated
    at the current time.

    Example
    -------
    >>> tspan = np.linspace(0, 5, 5)
    >>> t, y = euler(ball_motion, tspan, 50)
    >>> print(t)
        ndarray([0, 1, 2, 3, 4, 5])
    >>> print(y)
        ndarray([50.0000, -6.9239, -17.6275, -33.2847, -63.9675, -150.8969])

    """

    tspan = np.asarray(tspan)
    y0 = np.asarray(y0)

    # Determine the number of items in outputs
    num_times = tspan.shape[0]
    num_states = max(y0.size, 1)  # Don't use shape because it might be scalar

    # make sure odefun is compatible with y0
    f = odefun(tspan[0], y0)
    if f.shape != y0.shape:
        raise ValueError('odefun not compatible with y0')

    # Initialize outputs
    t = np.zeros(num_times)
    y = np.zeros((num_times, num_states))

    # Assign first row of outputs
    t[0] = tspan[0]
    y[0] = y0

    # Assign other rows of output
    for idx in range(num_times-1):

        # Calculate slopes at current time
        yprime = odefun(t[idx], y[idx])

        # Calculate next state
        dt = tspan[idx+1] - tspan[idx]
        y[idx+1] = y[idx] + yprime * dt
        t[idx+1] = tspan[idx+1]

    if num_states == 1:
        # Send flattened array
        y = y.reshape(t.shape)

    return t, y
