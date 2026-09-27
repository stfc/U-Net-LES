#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <tuple>
#include <string>

#include <bout/derivs.hxx>
#include <bout/invert_laplace.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>

#include <chrono>
using namespace std::chrono;

// Publish resolved BOUT options before callback import without changing the
// process environment. Python reads only these BOUT input settings.
void configureUnetOptions(bool implicit) {
  auto& options = Options::root()["unet"];
  PyObject* module = PyImport_AddModule("_pbout_unet_options"); // borrowed
  PyObject* configuration = module ? PyModule_GetDict(module) : nullptr; // borrowed
  if (!configuration) {
    PyErr_Print();
    throw BoutException("Could not create Python U-Net configuration");
  }
  Py_INCREF(configuration);

  const auto setOption = [configuration](const std::string& name,
                                       const std::string& value) {
    PyObject* text = PyUnicode_FromString(value.c_str());
    const int status = text ? PyMapping_SetItemString(configuration, name.c_str(), text) : -1;
    Py_XDECREF(text);
    if (status < 0) {
      PyErr_Print();
      throw BoutException("Could not set U-Net option " + name);
    }
  };

  try {
    // Absent entries retain the Python callback's checkpoint/default values.
    const char* settings[] = {
      "checkpoint",
      "dns_path",
      "dns_init",
      "dns_field_dir",
      "rescaling",
      "alpha",
      "reconstruct_channels",
      "device",
      "site_packages",
      "write_diagnostics",
      "omega_from_potential",
      "laplacian_order",
      "inverse_cutoff_fraction",
      "inverse_cutoff_order",
      "inverse_filter",
      "inverse_method",
      "inverse_min_wavenumber",
      "van_cittert_iterations",
      "van_cittert_relaxation",
      "post_cutoff_fraction",
      "post_cutoff_order",
      "post_filter",
    };
    for (const auto& setting : settings) {
      if (options[setting].isSet()) {
        setOption(setting, options[setting].as<std::string>());
      }
    }
    // Always use the same return convention in C++ and Python.
    setOption("implicit", implicit ? "1" : "0");
    auto& root = Options::root();
    const auto datadir = root["datadir"].withDefault(std::string("data"));
    const auto filename = root["optionfile"].withDefault(std::string("BOUT.inp"));
    setOption("bout_inp", !filename.empty() && filename.front() == '/' ? filename
                                                         : datadir + "/" + filename);
  } catch (...) {
    Py_DECREF(configuration);
    throw;
  }
  Py_DECREF(configuration);
}

void initPythonModule(PyObject **pModule, PyObject **pInitFlow,
                      PyObject **pFindLESTerms, PyObject **pSaveArakawaTerms,
                      bool implicit) {


  // Initialize Python interpreter
  Py_Initialize();
  configureUnetOptions(implicit);


  // set Python system path
  PyObject *sys_path = PySys_GetObject("path");

  PyObject *repr = PyObject_Repr(sys_path);  // sys_path -> string object
  //const char *str = PyUnicode_AsUTF8(repr);  // get UTF-8 C string
  //printf("current sys_path %s\n", str);
  Py_XDECREF(repr);

  // CMake resolves ../../ from this model's source directory, independent of cwd.
  PyObject *module_path = PyUnicode_FromString(HW_PYTHON_MODULE_DIR);
  if (module_path == NULL || PyList_Insert(sys_path, 0, module_path) < 0) {
    Py_XDECREF(module_path);
    PyErr_Print();
    throw BoutException("Could not add the pBOUT-unet.py directory to sys.path");
  }
  Py_DECREF(module_path);

  repr = PyObject_Repr(sys_path);
  //str = PyUnicode_AsUTF8(repr);
  //printf("modified sys_path %s\n", str);
  Py_XDECREF(repr);

  // Import Python module
  *pModule = PyImport_ImportModule("pBOUT-unet");
  if(*pModule == NULL) {
    PyErr_Print();
    throw BoutException("Could not import pBOUT-unet.py");
  }

  // The callback first adds [unet] site_packages to the embedded
  // interpreter's path. Only then can NumPy be imported from a virtualenv.
  if (_import_array() < 0) {
    PyErr_Print();
    throw BoutException("Could not initialize the NumPy C API");
  }

  // Load Python functions and execute it
  *pInitFlow = PyObject_GetAttrString(*pModule, "initFlow");

  if (!(*pInitFlow) || !PyCallable_Check(*pInitFlow)) {
    PyErr_Print();
    throw BoutException("Python initFlow function not found or not callable");
  }


  // Load Python functions and execute it
  *pFindLESTerms = PyObject_GetAttrString(*pModule, "findLESTerms");

  if (!(*pFindLESTerms) || !PyCallable_Check(*pFindLESTerms)) {
    PyErr_Print();
    throw BoutException("Python findLESTerms function not found or not callable");
  }

  *pSaveArakawaTerms = PyObject_GetAttrString(*pModule, "saveArakawaTerms");

  if (!(*pSaveArakawaTerms) || !PyCallable_Check(*pSaveArakawaTerms)) {
    PyErr_Print();
    throw BoutException("Python saveArakawaTerms function not found or not callable");
  }

  return;

}


void saveArakawaTerms(const int pStep, const double simtime,
                      const Field3D& phiVort, const Field3D& phiN,
                      PyObject *pSaveArakawaTerms) {
  const int NX = phiVort.getNx() - 4; // two guard cells on each x boundary
  const int NZ = phiVort.getNz();
  const int SIZE = 4 + 2 * NX * NZ;
  double *payload = new double[SIZE];

  payload[0] = double(pStep);
  payload[1] = simtime;
  payload[2] = double(NX);
  payload[3] = double(NZ);

  int cont = 4;
  for (int field = 0; field < 2; field++) {
    const Field3D& term = (field == 0) ? phiVort : phiN;
    for (int i = 2; i < phiVort.getNx() - 2; i++)
      for (int k = 0; k < NZ; k++)
        payload[cont++] = term(i, 0, k);
  }

  npy_intp dims[1]{SIZE};
  PyObject *array = PyArray_SimpleNewFromData(
      1, dims, NPY_DOUBLE, reinterpret_cast<void*>(payload));
  if (array == NULL) {
    PyErr_Print();
    delete [] payload;
    throw BoutException("Could not create the Arakawa diagnostic array");
  }

  PyObject *result = PyObject_CallFunctionObjArgs(
      pSaveArakawaTerms, array, NULL);
  if (result == NULL) {
    PyErr_Print();
    Py_DECREF(array);
    delete [] payload;
    throw BoutException("Call to Python saveArakawaTerms failed");
  }

  Py_DECREF(result);
  Py_DECREF(array);
  delete [] payload;
}








double* initFlow(double dx, double dy, Field3D n, Field3D phi, Field3D vort, PyObject *pModule, PyObject *pInitFlow) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZE  = n.getNz();
  const int SIZE2 = 2+3*SIZE*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZE2];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZE2};




  // pass n and vort arrays to pLES
  pLES[0] = dx;
  pLES[1] = dy;

  cont=2;
  for(i=2; i<SIZE+2; i++)   // we assume 2 guards cells in x-direction
    for(j=0; j<1; j++)
      for(k=0; k<SIZE; k++){
        pLES[cont++] = n(i,j,k);
        pLES[cont++] = phi(i,j,k);
        pLES[cont++] = vort(i,j,k);
      }



  // convert to numpy array   
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(pLES));
  if (pArray)
  {

    // create arguments
    pArgs = reinterpret_cast<PyArrayObject*>(pArray);
    if (pArgs!=NULL) {

      // call function
      pValue = PyObject_CallFunctionObjArgs(pInitFlow, pArray, NULL);

      if (pValue!=NULL) {

        pReturn = reinterpret_cast<PyArrayObject*>(pValue);
        printf("Dimensions of returned n array are: %d\n", PyArray_NDIM(pReturn));

        // convert result back to C++
        fLES = reinterpret_cast<double*>(PyArray_DATA(pReturn));

        // decrement Python object counter
        // Py_DECREF(pReturn);
        // PyErr_Print();
        // Py_XDECREF(pValue);
        // Py_DECREF(pArgs);
        // Py_DECREF(pArray);

      } else {
        PyErr_Print();
        Py_XDECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(pInitFlow);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call to Python function failed!\n");
      }
    
    } else {
      PyErr_Print();
      fprintf(stderr, "Arguments not created!\n");
    }

  } else {
    PyErr_Print();
    fprintf(stderr, "Array not created!\n");
  }


  delete [] pLES;

  return fLES;

}





double* findLESTerms(const int pStep, const double dx, const double simtime, const Field3D& n, Field3D& phi, Field3D& vort, Field3D& pPhiVort, Field3D& pPhiN,
  const bool implicitStylES, PyObject *pModule, PyObject *pFindLESTerms) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZE  = n.getNz();
  const int SIZE2 = 4+5*SIZE*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZE2];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZE2};



  // pass n and vort arrays to pLES
  pLES[0] = double(pStep);
  pLES[1] = 0.0; // Reserved header slot; Python reads dns_init from BOUT options.
  pLES[2] = dx;
  pLES[3] = simtime;

  cont=4;
  if (implicitStylES){
    int N_LES = n.getNz();
    for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
      for(int j=0; j<1; j++)
        for(int k=0; k<n.getNz(); k++){
          pLES[cont + 0*N_LES*N_LES] = n(i,j,k);
          pLES[cont + 1*N_LES*N_LES] = phi(i,j,k);
          pLES[cont + 2*N_LES*N_LES] = vort(i,j,k);
          pLES[cont + 3*N_LES*N_LES] = pPhiVort(i,j,k);
          pLES[cont + 4*N_LES*N_LES] = pPhiN(i,j,k);                
          cont = cont+1;
        }
  }
  else{
    int N_LES = n.getNz();
    for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
      for(int j=0; j<1; j++)
        for(int k=0; k<n.getNz(); k++){
          pLES[cont + 0*N_LES*N_LES] = n(i,j,k);
          pLES[cont + 1*N_LES*N_LES] = phi(i,j,k);
          pLES[cont + 2*N_LES*N_LES] = vort(i,j,k);
          cont = cont+1;
        }
  }

  // convert to numpy array   
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(pLES));
  if (pArray)
  {

    // create arguments
    pArgs = reinterpret_cast<PyArrayObject*>(pArray);
    if (pArgs!=NULL) {

      // call function
      pValue = PyObject_CallFunctionObjArgs(pFindLESTerms, pArray, NULL);

      if (pValue!=NULL) {

        pReturn = reinterpret_cast<PyArrayObject*>(pValue);
        //printf("Dimensions of returned n array are: %d\n", PyArray_NDIM(pReturn));

        // convert result back to C++
        fLES = reinterpret_cast<double*>(PyArray_DATA(pReturn));

        // decrement Python object counter
        // Py_DECREF(pReturn);
        // PyErr_Print();
        // Py_XDECREF(pValue);
        // Py_DECREF(pArgs);
        // Py_DECREF(pArray);

      } else {
        PyErr_Print();
        Py_XDECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(pFindLESTerms);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call to Python function failed!\n");
      }
    
    } else {
      PyErr_Print();
      fprintf(stderr, "Arguments not created!\n");
    }

  } else {
    PyErr_Print();
    fprintf(stderr, "Array not created!\n");
  }


  delete [] pLES;

  return fLES;

}






void closePythonModule(PyObject **pModule, PyObject **pInitFlow,
                       PyObject **pFindLESTerms, PyObject **pSaveArakawaTerms) {

  // shutdown Python interpreter
  Py_DECREF(pInitFlow);
  Py_DECREF(pFindLESTerms);
  Py_DECREF(pSaveArakawaTerms);
  Py_DECREF(pModule);

  if (Py_FinalizeEx() < 0) {
    PyErr_Print();
    fprintf(stderr, "Failed to shutdown Python!");
  }

  return;

}







class HW : public PhysicsModel {
private:
  Field3D n, vort; // Evolving density and vorticity
  Field3D phi;     // Electrostatic potential

  // Python variables
  PyObject *pModule;
  PyObject *pInitFlow;
  PyObject *pFindLESTerms;
  PyObject *pSaveArakawaTerms;

  int pStep      = 0;
  int dns_init = 0;  // dns_init: -1=fresh DNS, 2=DNS from fields; both use native brackets.

  double deltax;
  double deltaz;
  double psimtime = 0.0;

  bool profile_StylES = false;
  bool implicitStylES = true;
  bool writeArakawaTerms = false;

  Field3D pPhiVort;
  Field3D pPhiN;

  Field3D tPhiVort;
  Field3D tPhiN;
  

  int totCount = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> timeStop;
  std::chrono::microseconds totDuration1{0};
  std::chrono::microseconds totDuration2{0};

  // Model parameters
  BoutReal alpha;     // Adiabaticity (~conductivity)
  BoutReal kappa;     // Density gradient drive
  BoutReal Dvort, Dn; // Diffusion
  bool modified;      // Modified H-W equations?

  // Poisson brackets: b0 x Grad(f) dot Grad(g) / B = [f, g]
  // Method to use: BRACKET_ARAKAWA, BRACKET_STD or BRACKET_SIMPLE
  BRACKET_METHOD bm; // Bracket method for advection terms

  std::unique_ptr<Laplacian> phiSolver; // Laplacian solver for vort -> phi

  // Simple implementation of 4th order perpendicular Laplacian
  Field3D Delp4(const Field3D& var) {
    Field3D tmp;
    tmp = Delp2(var);
    mesh->communicate(tmp);
    tmp.applyBoundary("neumann");
    return Delp2(tmp);

    //return Delp2(var);
  }

protected:
  int init(bool UNUSED(restart)) {

    auto& options = Options::root()["hw"];
    alpha = options["alpha"].withDefault(1.0);
    kappa = options["kappa"].withDefault(0.1);
    Dvort = options["Dvort"].withDefault(1e-2);
    Dn = options["Dn"].withDefault(1e-2);

    modified = options["modified"].withDefault(false);
    auto& unet = Options::root()["unet"];
    dns_init = unet["dns_init"].withDefault(0);
    if (dns_init < -1 || dns_init > 3) {
      throw BoutException("dns_init must be -1, 0, 1, 2, or 3");
    }
    if (dns_init == -1) {
      // Python-only settings may remain in a shared input file for fresh DNS.
      unet.setConditionallyUsed();
    }
    writeArakawaTerms = unet["write_arakawa_terms"].isSet()
        ? unet["write_arakawa_terms"].as<bool>()
        : options["write_arakawa_terms"].withDefault(false);
    implicitStylES = unet["implicit"].withDefault(true);
    profile_StylES = unet["profile"].withDefault(false);

    SOLVE_FOR(n, vort);
    SAVE_REPEAT(phi);

    // Split into convective and diffusive parts
    setSplitOperator();

    phiSolver = Laplacian::create();
    phi = 0.; // Starting phi
    
    pPhiVort = 0.;
    pPhiN = 0.;

    tPhiVort = 0.;
    tPhiN = 0.;



    if (dns_init != -1) {
      initPythonModule(&pModule, &pInitFlow, &pFindLESTerms,
                       &pSaveArakawaTerms, implicitStylES);
    }

    CELL_LOC outloc = n.getLocation();
    Coordinates *metric = phi.getCoordinates(outloc);

    deltax = metric->dx(0,0,0);
    deltaz = metric->dz(0,0,0);

    // Native DNS starts at t=0 with BOUT.inp initial conditions.
    // Mode 2 and LES/filter/replay modes still import fields through Python.
    if (dns_init != -1) {
      double* npv = initFlow(deltax, deltaz, n, phi, vort, pModule, pInitFlow);
      const int N_LES = n.getNz();
      int cont = 0;
      for (int i = 2; i < n.getNx() - 2; ++i)
        for (int j = 0; j < 1; ++j)
          for (int k = 0; k < n.getNz(); ++k) {
            n(i,j,k) = npv[cont + 0*N_LES*N_LES];
            phi(i,j,k) = npv[cont + 1*N_LES*N_LES];
            vort(i,j,k) = npv[cont + 2*N_LES*N_LES];
            ++cont;
          }
    }

    // // close Python console
    // closePythonModule(&pModule, &pInitFlow, &pFindLESTerms);


    // Communicate variables
    mesh->communicate(n, phi, vort);


    // Use default flags

    // Choose method to use for Poisson bracket advection terms
    switch (options["bracket"].withDefault(0)) {
    case 0: {
      bm = BRACKET_STD;
      output << "\tBrackets: default differencing\n";
      break;
    }
    case 1: {
      bm = BRACKET_SIMPLE;
      output << "\tBrackets: simplified operator\n";
      break;
    }
    case 2: {
      bm = BRACKET_ARAKAWA;
      output << "\tBrackets: Arakawa scheme\n";
      break;
    }
    case 3: {
      bm = BRACKET_CTU;
      output << "\tBrackets: Corner Transport Upwind method\n";
      break;
    }
    default:
      output << "ERROR: Invalid choice of bracket method. Must be 0 - 3\n";
      return 1;
    }

    return 0;
  }




  int convective(BoutReal time) {
    // Non-stiff, convective part of the problem

    // Solve for potential
    if (pStep>0 || dns_init == -1){
      phi = phiSolver->solve(vort, phi);
    }

    double *rLES;

    // Communicate variables
    mesh->communicate(n, vort, phi);

    if (writeArakawaTerms && dns_init != -1) {
      Field3D arakawaPhiVort = bracket(phi, vort, bm);
      Field3D arakawaPhiN = bracket(phi, n, bm);
      saveArakawaTerms(pStep, time, arakawaPhiVort, arakawaPhiN,
                       pSaveArakawaTerms);
    }

    if (pStep==0){
      psimtime = time;
    }

    // find brackets term (for explicit) or subgrid scale terms (for implicit) via StylES
    if (dns_init != -1 && dns_init != 2)
    {
      if (implicitStylES){
        pPhiVort = bracket(phi, vort, bm);
        pPhiN    = bracket(phi, n, bm);
      }
      double simtime = time;
      output_progress.print("\r");
      rLES = findLESTerms(pStep, deltax, simtime, n, phi, vort, pPhiVort, pPhiN, implicitStylES, pModule, pFindLESTerms);
      int N_LES = n.getNz();
      int LES_it = int(rLES[0]);
      int cont=1;

      if (implicitStylES){
        for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
          for(int j=0; j<1; j++)
            for(int k=0; k<n.getNz(); k++){
              tPhiVort(i,j,k) = rLES[cont + 0*N_LES*N_LES];
              tPhiN(i,j,k)    = rLES[cont + 1*N_LES*N_LES];
              cont = cont+1;
            }
      }
      else{
        for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
          for(int j=0; j<1; j++)
            for(int k=0; k<n.getNz(); k++){
              pPhiVort(i,j,k) = rLES[cont + 0*N_LES*N_LES];
              pPhiN(i,j,k)    = rLES[cont + 1*N_LES*N_LES];
              cont = cont+1;
            }
      }
    }
    else{
      pPhiVort = bracket(phi, vort, bm);
      pPhiN    = bracket(phi, n, bm);
    }


    // Modified H-W equations, with zonal component subtracted from resistive coupling term
    if (profile_StylES){
      timeStart = high_resolution_clock::now();
    }


    Field3D nonzonal_n = n;
    Field3D nonzonal_phi = phi;
    if (modified) {
      // Subtract average in Y and Z
      nonzonal_n -= averageY(DC(n));
      nonzonal_phi -= averageY(DC(phi));
    }

    ddt(vort) = -pPhiVort + alpha*(nonzonal_phi - nonzonal_n);
    ddt(n)    = -pPhiN    + alpha*(nonzonal_phi - nonzonal_n) - kappa*DDZ(phi);

    if (profile_StylES)
    {
      timeStop = high_resolution_clock::now();
      auto duration1 = duration_cast<microseconds>(timeStop - timeStart);
      totDuration1 = totDuration1 + duration1;
      printf("timing convective all %lld\n", totDuration2.count());
    }

    return 0;
  }
  



  int diffusive(BoutReal UNUSED(time)) {

    if (profile_StylES)
    {
      timeStart = high_resolution_clock::now();
    }

    // Diffusive terms
    mesh->communicate(n, vort);

    if (implicitStylES){
      ddt(vort) = - Dvort*Delp4(vort) + tPhiVort;
      ddt(n)    = - Dn*Delp4(n)       + tPhiN;
    }
    else{
      ddt(vort) = - Dvort*Delp4(vort);
      ddt(n)    = - Dn*Delp4(n);
    }


    if (profile_StylES)
    {
      timeStop = high_resolution_clock::now();
      auto duration2 = duration_cast<microseconds>(timeStop - timeStart);
      totDuration2 = totDuration2 + duration2;
      totCount=totCount+1;
      printf("timing diffusion %lld  %d\n", totDuration2.count(), totCount);
    }

    pStep++;
    return 0;
  }

};


// Define a main() function
BOUTMAIN(HW);
