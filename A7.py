# Julian Castrillon
# CFD - Spring 2022
# A demo FEniCS script for solving the unsteady form of Navier-Stokes equations
# using Q1P1 elements with streamwise Upwind Petrov-Galerkin stabilization for
# convection instabilities, and Petrov Galerkin Pressure Stabilzation for linear
# velocity-pressure combination.

# import sys
import numpy as np
import fenics as fe

#################################################################################################
## Definition of some global FEniCS optimization parameters #####################################

fe.set_log_level(fe.LogLevel.INFO)
fe.parameters['form_compiler']['representation']    = 'uflacs'
fe.parameters['form_compiler']['optimize']          = True
fe.parameters['form_compiler']['cpp_optimize']      = True
fe.parameters["form_compiler"]["cpp_optimize_flags"]= '-O2 -funroll-loops'

#################################################################################################
## Parameters ###################################################################################

meshFile  = "A7.xml" # Input mesh
facetFile = "A7_facet_region.xml" # Input mesh boundaries
outFileV  = "ResultsQ2Q1/Vel.pvd"
outFileP  = "ResultsQ2Q1/Pres.pvd"
D         = 0.2    # Diameter
U0        = 1.5    # Inlet Velocity fe.Expression['']
dt        = 0.01   # Timestep
t_end     = 30     # Length of simulation
mesh      = fe.Mesh(meshFile) # Create the mesh and import mesh into the solver
h         = fe.CellDiameter(mesh) # Mesh size

# Parameters defined in FEniCS compatible syntax
mu    = fe.Constant(1/500)     # Viscosity
idt   = fe.Constant(1/dt)      # Inverse time step
theta = fe.Constant(0.5)       # Crank-Nicholson time-stepping scheme
b     = fe.Constant((0,0))     # Body forces
n     = fe.FacetNormal(mesh)   # Normal vector  

# Define the mixed vector function space operating on this meshed domain
V  = fe.VectorElement('Lagrange', mesh.ufl_cell(), 2)
P  = fe.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
M  = fe.MixedElement([V, P])
FS = fe.FunctionSpace(mesh, M) # Function space
TFS    = fe.TrialFunction(FS)  # Trial function
(v, q) = fe.TestFunctions(FS)  # Define test functions

# Defining essential/Dirichlet boundary conditions
# Step 1: Identify all boundary segments forming Gamma_d
domainBoundaries = fe.MeshFunction("size_t", mesh, facetFile)
ds               = fe.ds(subdomain_data=domainBoundaries)

#################################################################################################
## Boundary Conditions ##########################################################################

# Identification of all correct boundary markers needed for the domain
ID_Inlet  = 1
ID_Top    = 2
ID_Outlet = 3
ID_Bottom = 4
ID_Circle = 5

NoSlip    = fe.Constant((0,0))
POut      = fe.Constant(0)
InletFlow = fe.Constant((U0,0))

InletBC  = fe.DirichletBC(FS.sub(0),InletFlow,domainBoundaries,ID_Inlet)
TopBC    = fe.DirichletBC(FS.sub(0),NoSlip,   domainBoundaries,ID_Top)
OutletBC = fe.DirichletBC(FS.sub(1),POut ,    domainBoundaries,ID_Outlet)
BottomBC = fe.DirichletBC(FS.sub(0),NoSlip,   domainBoundaries,ID_Bottom)
CircleBC = fe.DirichletBC(FS.sub(0),NoSlip,   domainBoundaries,ID_Circle)

bcs = [InletBC,OutletBC,TopBC,BottomBC,CircleBC]

## Boundary check
# fe.File('New.pvd') << domainBoundaries
# sys.exit()

#################################################################################################
## Theta - Galerkin formulation #################################################################

w = fe.Function(FS) # Timestep n+1

(u,p) = (fe.as_vector((w[0],w[1])),w[2])
T1_1 = fe.inner(v,fe.grad(u)*u)*fe.dx
T2_1 = mu*fe.inner(fe.grad(v), fe.grad(u))*fe.dx
T3_1 = p*fe.div(v)*fe.dx
T4_1 = q*fe.div(u)*fe.dx
T5_1 = fe.dot(v,b)*fe.dx
T_1 = T1_1+T2_1-T3_1-T4_1-T5_1

w0 = fe.Function(FS) # Timestep n

(u0,p0) = (fe.as_vector((w0[0],w0[1])),w0[2])
T1_0 = fe.inner(v,fe.grad(u0)*u0)*fe.dx
T2_0 = mu*fe.inner(fe.grad(v),fe.grad(u0))*fe.dx
T3_0 = p*fe.div(v)*fe.dx
T4_0 = q*fe.div(u0)*fe.dx
T5_0 = fe.dot(v,b)*fe.dx
T_0 = T1_0+T2_0-T3_0-T4_0-T5_0

F = idt*fe.inner((u-u0),v)*fe.dx+(theta)*T_1+(1-theta)*T_0 #(u-u0)/dt + (1-theta)*F(n) + theta*F(n+1) = 0

#################################################################################################
## Streamwise Upwind Petrov Galerkin (SUPG) stabilization for convection ######################## 

vnorm = fe.sqrt(fe.dot(u0,u0))
tau = ((2*theta*idt)**2+(2*vnorm/h)**2+(4*mu/h**2)**2)**(-0.5)

# Residual of the strong form of Navier-Stokes and continuity
R = idt*(u-u0)+ theta*(fe.grad(u)*u-mu*fe.div(fe.grad(u))+fe.grad(p)-b) + (1-theta)*(fe.grad(u0)*u0-mu*fe.div(fe.grad(u0))+fe.grad(p)-b)

# Streamwise Upwind Petrov Galerkin (SUPG) stabilization for convection
F += tau*fe.inner(fe.grad(v)*u,R)*fe.dx(metadata={'quadrature_degree':4})

# Petrov Galerkin Pressure Stabilzation (PSPG) stabilization for pressure field // The Ladyzhenskaya-Babuska-Brezzi condition
# F += -tau*fe.inner(fe.grad(q),R)*fe.dx(metadata={'quadrature_degree':4})

# Jacobian for Newton method
Jacobian = fe.derivative(F,w,TFS)

#################################################################################################
## Solver #######################################################################################

problem = fe.NonlinearVariationalProblem(F,w,bcs,J=Jacobian)
solver  = fe.NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 20

# Create files for storing solution
ufile = fe.File(outFileV)
pfile = fe.File(outFileP)

# Loops #####################################
# Inner loop - Newton iteration at each step
# Outer loop - Time steps

t   = dt
tn  = 0
forces = [] # Initiate array to log data

while t < t_end:

    print("t = ",np.around(t,3))
    print("Solving ....")
    solver.solve()
    (u,p) = w.split()

    ## Calculate Lift and Drag ####
    sigma    = -p*fe.Identity(2)+2*mu*(1/2*(fe.grad(u)+fe.grad(u).T))
    traction = fe.dot(sigma, n)
    forceX   = traction[0]*ds(ID_Circle)
    forceY   = traction[1]*ds(ID_Circle)
    drag     = fe.assemble(forceX) # Fenics integration (fe.assmble)
    lift     = fe.assemble(forceY) # Fenics integration (fe.assmble)
    Cd       = 2*drag/((2/3*U0)**2*D) # Drag coefficient
    Cl       = 2*lift/((2/3*U0)**2*D) # Lift coefficient
    forces.append([drag,lift,Cd,Cl])
    
    if tn%4 == 0:   # Save to file only if the time step increases by a step of 4 
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        ufile << u
        pfile << p
        print("Written Velocity And Pressure Data")
    
    w0.assign(w)
    t   += dt
    tn  += 1

np.savetxt('ForcesQ2Q1.txt',forces)
#################################################################################################
## END ##########################################################################################
#################################################################################################