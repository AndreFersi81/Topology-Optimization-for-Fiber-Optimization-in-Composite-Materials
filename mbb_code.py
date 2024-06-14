from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define the domain dimension
height = 100
width = 300

# Define the approximate size of the element
elsize = 10

# Initialize alpha
alpha = 1e-2

# Define the load
t_ = 15

# Increment for alpha
delta_alpha = 1e-2

# Reduction factor
alpha_red = 0.5

# Tolerance for the bisection method convergence
tol = 1e-3

# Tolerance for the steepest descent convergence
eps = 1e-4

# Material properties
E1 = Constant(38.6e3)  # MPa - Young's modulus in fiber direction
E2 = Constant(8.17e3)  # MPa - Young's modulus in isotropic plane
G12 = Constant(4.14e3)  # MPa - Shear modulus
nu12 = Constant(0.26)  # Poisson ratio

# Define the load as a Constant vector
t = Constant((0, -t_))

# Define the number of elements
nx = int(width / elsize)
ny = int(height / elsize)

# Create the mesh
mesh = RectangleMesh(Point(0, 0), Point(width, height), nx, ny, 'crossed')

# Define space functions
V = VectorFunctionSpace(mesh, 'CG', 1)
P = FunctionSpace(mesh, 'DG', 0)

# Define the initial variables
phi = Function(P, name='Fiber angle')
u = Function(V, name='Displacement')
u_ = TrialFunction(V)
v = TestFunction(V)
phi_inc = Function(P, name='Fiber angle + delta_phi * DLa(phi)')
DL_phi = Function(P, name='DL_phi')


class Clamped(SubDomain):
    '''
    Clamped boundary coordinates
    '''

    def inside(self, x, on_boundary):
        '''
        '''
        return between(x[0], (0., 10.)) and near(x[1], 0) and on_boundary


class Slide(SubDomain):
    '''
    Slide boundary coordinates
    '''

    def inside(self, x, on_boundary):
        '''
        '''
        return between(x[0], (width - 10, width)
                       ) and near(x[1], 0) and on_boundary


class Load(SubDomain):
    '''
    Load coordinates
    '''

    def inside(self, x, on_boundary):
        '''
        '''
        return near(x[1], height) and \
            between(x[0], (0.5 * width - 10, 0.5 * width + 10)) and \
            on_boundary


# Create a mesh function to apply the boundary conditions
facets = MeshFunction("size_t", mesh, 1)

# Create dx and ds
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Setting all mesh as 0
facets.set_all(0)

# Defining different colors for Clamped, Slide and Load
Clamped().mark(facets, 1)
Load().mark(facets, 2)
Slide().mark(facets, 3)

# Create the Dirichlet boundary conditions
bc1 = DirichletBC(V, ((0., 0.)), facets, 1)
bc2 = DirichletBC(V.sub(1), 0., facets, 3)
bc = [bc1, bc2]


def strain(u):
    '''
    Calculate the strain in Voigt notation.
    '''

    eps_x = u[0].dx(0)
    eps_y = u[1].dx(1)
    gamma_xy = 0.5 * (u[0].dx(1) + u[1].dx(0))

    return as_vector((eps_x, eps_y, gamma_xy))


def reuter_matrix():
    '''
    Reuter matrix used to calculate the orthotropic
    constitutive tensor
'''
    return as_tensor(((1, 0, 0), (0, 1, 0), (0, 0, 2)))


def transformation_matrix(phi):
    '''
    Matrix that transforms the coordinates.
    '''

    # Cossine of c
    c = cos(phi)

    # Sine of c
    s = sin(phi)

    # Tranformation matrix
    T = as_tensor((
        (c*c, s*s,  2*c*s),
        (s*s, c*c, -2*s*c),
        (-s*c, s*c,  c*c-s*s)))

    return T


def orthotropic_constitutive_tensor(E1, E2, G12, nu12):
    '''
    Calculate the orthotropic constitutive tensor
    '''
    # Tensor S
    S11 = 1. / E1
    S12 = -nu12 / E1
    S22 = 1. / E2
    S66 = 1. / G12

    # Constitutive tensor elements
    Q11 = S22 / (S11 * S22 - S12**2)
    Q12 = - S12 / (S11 * S22 - S12**2)
    Q22 = S11 / (S11 * S22 - S12**2)
    Q66 = 1. / S66

    # Constitutive tensor
    Q = as_tensor(((Q11, Q12, 0),
                  (Q12, Q22, 0),
                  (0,   0, Q66)))

    return Q


def a(phi, u, v):
    '''
    Bilinear expression for the weak formulation of the problem.
    '''

    # Define the tensor to calculate the constitutive tensor C
    R = reuter_matrix()
    Q = orthotropic_constitutive_tensor(E1, E2, G12, nu12)
    T = transformation_matrix(phi)

    # Calculate the constitutive tensor C
    C = inv(T) * Q * R * T * inv(R)

    return inner(C * strain(u), strain(v)) * dx


def L(v):
    '''
    Linear expression for the weak formulation of the problem.
    '''
    return inner(t, v) * ds(2)


def F(phi, u, v, u_):
    '''
    Solve the bilinear problem for u.
    '''

    # Assemble the system
    A, b = assemble_system(a(phi, u_, v), L(v), bc)

    # Solve the system for u
    solve(A, u.vector(), b)

    return u


def compliance(phi, u):
    '''
    '''
    return assemble(a(phi, u, u), annotate=True)


def DC_phi(phi):
    '''
    '''

    # Orthotropic constitutive tensor
    Q = orthotropic_constitutive_tensor(E1, E2, G12, nu12)
    Q11 = Q[0, 0]
    Q12 = Q[0, 1]
    Q22 = Q[1, 1]
    Q66 = Q[2, 2]

    # Components of DC_phi
    c = cos(phi)
    s = sin(phi)
    c2 = c**2
    s2 = s**2
    c4 = cos(4*phi)
    s4 = sin(4*phi)
    Delta_a = Q11 - Q12 - 2*Q66
    Delta_b = Q12 - Q22 + 2*Q66
    Delta_c = Q11/2 - Q12/2 + Q22/2 - 2*Q66

    DC_phi11 = -4 * (Q11*c2 - Q22*s2 + (Q12 + 2*Q66)*s2 - (Q12 + 2*Q66)*c2
                     ) * s * c
    DC_phi12 = Delta_c * s4
    DC_phi16 = 3 * (c4 - 1) * Delta_a/8 + 3 * (c4 - 1) * Delta_b/8 + Delta_c *\
        c2**2 + Delta_b * s2**2
    DC_phi22 = 4 * (Q11*s2 - Q22*c2 - (Q12 + 2*Q66)*s2 + (Q12 + 2*Q66)*c2) * s\
        * c
    DC_phi26 = 3 * (c4 - 1) * (-Delta_a)/8 + 3 * (c4 - 1) * Delta_b/8 + (
        -Delta_a * s2**2 + Delta_b * c2**2)
    DC_phi66 = Delta_c * s4

    # Derivative of the constitutive tensor
    return as_tensor([[DC_phi11, DC_phi12, DC_phi16],
                      [DC_phi12, DC_phi22, DC_phi26],
                      [DC_phi16, DC_phi26, DC_phi66]])


# Create *.xdmf
xdmf_file = XDMFFile('fields/fields_cg_mbb.xdmf')

# Initializate c list
J_list = []

# Set some parameters
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

# Initializate convergence variable
conv = False

# Initializate max_diff
max_diff = '-'

# Initializate d_old
d_old = np.zeros(P.dim())
DL_phi_old = np.zeros(P.dim())

# Calculate the field of the area of elements
delta_rho = project(CellVolume(mesh), P).vector().get_local()

while not conv:

    # Calculate the displacements
    u = F(phi, u, v, u_)

    # Calculate the gradient field
    DL_phi_new = project(inner(DC_phi(phi) * strain(u), strain(u)), P).vector(
    ).get_local() * delta_rho

    # Initialize alpha_best
    alpha_best = alpha

    # Old compliance (first compliance in this casa)
    J_old = compliance(phi, u)
    J_new = J_old

    DL_phi_old[:] = DL_phi_new[:]

    # Update phi new
    while delta_alpha > tol:

        # Calculate the new phi with an increment
        phi_inc.assign(phi)
        phi_inc.vector()[:] += alpha * DL_phi_new[:]

        # New compliance
        J_new = compliance(phi_inc, u)

        if J_new < J_old:

            # Update J_old
            J_old = J_new

            # Best alpha
            alpha_best = alpha

        else:

            # Update delta_alpha
            delta_alpha *= alpha_red

            # Update alpha
            alpha = alpha_best

        # Update alpha
        alpha += delta_alpha

    # Update phi_old
    phi_old_array = phi.vector().get_local()

    # Update phi
    phi.vector()[:] += alpha_best * DL_phi_new[:]

    # Save the fields of fiber angle and displacements
    if len(J_list) % 10 == 0:
        xdmf_file.write(phi, len(J_list))
        xdmf_file.write(u, len(J_list))
        xdmf_file.write(DL_phi, len(J_list))

    # Append c to the list
    J_list.append(J_new)

    # Print the iteration and objective function
    if len(J_list) <= 11:
        print('i: {:3d} - J: {:.6f} - max_diff: {}'.format(
            len(J_list), J_new, max_diff))
    else:
        print('i: {:3d} - J: {:.6f} - max_diff: {:.6e}'.format(
            len(J_list), J_new, max_diff))

    if len(J_list) > 10:

        # Calculate the convergence
        max_diff = abs(
            (np.array(J_list[-10:]) - np.array(J_list[-11:-1])).max())
        conv = max_diff < eps


print('i: {:3d} - J: {:.6f} - max_diff: {:.6e}'.format(
    len(J_list), J_new, max_diff))

# Plot the convergence of the objective function
plt.plot(range(len(J_list[1:])), J_list[1:], color='black')
plt.xlabel('Iterations')
plt.ylabel('Objective function')
plt.grid()
plt.savefig('convergence_gc_mbb.png')
