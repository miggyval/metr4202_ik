PLOT_DATA = True

# Import the modern robotics library with shorthand mr
import modern_robotics as mr
# Import the numpy library with shorthand np
import numpy as np
# Import pyplot with shorthand plt
if PLOT_DATA:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['text.usetex'] = True

# Import everything from the math library
from math import *
# Define helpful functions at the start, from the linear algebra library
norm = np.linalg.norm
pinv = np.linalg.pinv



# Define a reasonable tolerance for algorithms
tol = 1e-8

# Define parameters for robot dimensions
L1 = 425e-3
L2 = 392.25e-3
W1 = 135.85e-3
W2 = 93e-3
H1 = 89.159e-3
H2 = 94.65e-3

def plot_data(data):
    plt.plot(data[:, 6:8])
    plt.title("Error of Newton-Raphson Algorithm")
    plt.xlabel("Num. Iterations")
    plt.ylabel("Error (rad, m)")
    plt.legend([r"$\omega_{s}$", r"$v_{s}$"])
    plt.figure()
    plt.plot(data[:, :6])
    plt.title("Angles of Newton-Raphson Algorithm")
    plt.xlabel("Num. Iterations")
    plt.ylabel("Angles (rad)")
    plt.legend([r"$\theta_{1}$", r"$\theta_{2}$", r"$\theta_{3}$", r"$\theta_{4}$", r"$\theta_{5}$", r"$\theta_{6}$"])
    plt.show()

def MatrixLog6(T):
    """
    Calculates the logarithm of the transformation T
    log: SE3 -> se3
    """
    # Unpack the T matrix into rotation R and translation p
    R, p = mr.TransToRp(T)
    # Convert p into a column vector
    p_vec = np.array([p]).T
    # Calculate the non-normalized matrix [w]
    w_mat = mr.MatrixLog3(R)
    # Convert the non-normalized matrix into a vector w
    w_vec = mr.so3ToVec(w_mat)
    
    if norm(w_vec) < tol:
        # If the magnitude of w is less than a tolerance,
        # Calculate theta from the norm of translation vector p.
        # Set [w] to zero and set G_inv to I
        # ||w|| < 0 => [w_hat] = 0,
        #                theta = ||p||
        #                G_inv = I
        theta = norm(p_vec)
        w_hat_mat = np.zeros((3, 3))
        G_inv = np.eye(3)
    else:
        # Otherwse, calculate theta from the norm of w
        # Calculate the normalized [w_hat] matrix by dividing by theta
        # Calculate G_inv from the formula:
        #   theta = ||w||
        # [w_hat] = [w] / theta
        #   G_inv = I / theta - [w_hat] / 2 + (1 / theta - (cot(theta / 2) / 2) * [w_hat] ^ 2
        theta = norm(w_vec)
        w_hat_mat = w_mat / theta
        G_inv = np.eye(3) / theta - w_hat_mat / 2.0 + \
            (1.0 / theta - 1.0 / (2.0 * tan(theta / 2.0))) * (w_hat_mat @ w_hat_mat)

    # Finally, calculate v_hat
    v_hat_vec = G_inv @ p_vec
    # Return the non-normalized representations [w] and v
    w_mat = w_hat_mat * theta
    v_vec = v_hat_vec * theta
    # Return the 4x4 matrix:
    # [ [w] v  ]
    # [  0  0  ]
    return np.vstack((np.hstack((w_mat, v_vec)), np.zeros((1, 4))))

def IKinSpace(Slist, M, Tsd, thetalist0, eps_w, eps_v, maxiterations=20, get_data=False):
    """
    Calculates the inverse kinematics solution numerically
    Using the Newton-Raphson Method in the space frame
    """
    # Initialize the array of theta
    thetalist = np.array(thetalist0).copy()
    if get_data:
        data = np.zeros(shape=(maxiterations, 8))
    for i in range(maxiterations):
        # Calculate the current transformation given theta
        Tsb = mr.FKinSpace(M, Slist, thetalist)
        # Calculate the 'error screw' in the body frame
        # [Vb] = Tsb^-1 * Tsd
        Vb = mr.se3ToVec(MatrixLog6(mr.TransInv(Tsb) @ Tsd))
        # Use the adjoint representation of Tsb
        # to transform the error into the space frame
        Ad_Tsb = mr.Adjoint(Tsb)
        Vs = Ad_Tsb @ Vb
        # Decouple the linear and angular error
        ws = np.array([Vs[0], Vs[1], Vs[2]])
        vs = np.array([Vs[3], Vs[4], Vs[5]])
        # If the error is less than the tolerance, break.
        err = norm(ws) > eps_w or norm(vs) > eps_v
        if get_data:
            data[i, :6] = thetalist
            data[i, 6:] = np.array([norm(ws), norm(vs)])
        if not err:
            if get_data:
                data = data[:i,:]
            break
        # Calculate the Jacobian and adjust theta, based on the error
        Js = mr.JacobianSpace(Slist, thetalist)
        thetalist += pinv(Js) @ Vs
    if get_data:
        return (thetalist, not err, data)
    else:
        return (thetalist, not err, None)

# Define end-effector pose in zero configuration
M = np.array([
    [-1, 0, 0, L1 + L2],
    [ 0, 0, 1, W1 + W2],
    [ 0, 1, 0, H1 - H2],
    [ 0, 0, 0,       1]
])

# Define screw axes in the space frame
S1 = np.array([ 0, 0,  1,       0,       0,       0 ])
S2 = np.array([ 0, 1,  0,     -H1,       0,       0 ])
S3 = np.array([ 0, 1,  0,     -H1,       0,      L1 ])
S4 = np.array([ 0, 1,  0,     -H1,       0, L1 + L2 ])
S5 = np.array([ 0, 0, -1,     -W1, L1 + L2,       0 ])
S6 = np.array([ 0, 1,  0, H2 - H1,       0, L1 + L2 ])

# Create list of screw axes
Slist = np.array([S1, S2, S3, S4, S5, S6])
# Create test list of theta to generate desired pose
thetalist = pi * (np.random.random((6,)) - 0.5)
thetalist[0] = 1.0
print("Desired Angles:")
print(thetalist)
# Use forward kinematics to calculate desired pose
Tsd = mr.FKinSpace(M, Slist, thetalist)
# Generate initial list of theta
thetalist0 = np.array([1.0 - pi, 0.5, -0.1, 0.5, -0.4, 0.3])

# Use the NR method to calculate solution theta
thetalistsol, success, data = IKinSpace(Slist, M, Tsd, thetalist0, 1e-3, 1e-3, maxiterations=100, get_data=PLOT_DATA)
print("Actual Angles:")
print(thetalistsol)
# Use forward kinematics to calculate pose from solution theta
Tsol = mr.FKinSpace(M, Slist, thetalistsol)
# If the algorithm returned a success, print the transformation matrices

print("Desired Pose:")
print(Tsd)
print("Actual Pose:")
print(Tsol)
if success:
    print("Success!")
else:
    print("Failed to reach pose...")

if PLOT_DATA:
    plot_data(data)
