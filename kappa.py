import numpy, pickle, os, hashlib

# import bits from pyatomdb.
from pyatomdb import apec, util, const, atomdb, pyfits, spectrum

"""
This modules is designed to generate kappa specta.
Method is:
  - look up the coefficients from the Hahn & Savin paper,
  - Use these to assemble ionization and recombination rates
  - Solve ion balance
  - Make spectra for the relevant ions at temperatures defined by Hahn & Savin
  - Sum spectra to get... well, a spectrum.

"""

# set version number

# March 6th 2020, ARF:
#__version__='1.1.0'

# October 23rd 2020 ARF: Fixed bug caused by updates to pyatomdb
__version__='1.1.1'



class hs_data():
  """
  Class to store the read in Hahn & Savin Data.
  Can then be queried to spit out the relevant rates, etc.

  """

  def __init__(self,hsdatafile):
    """
    Read in the data
    """

    # for now, this is hardwired to load a pickle file. This is not
    # ideal, will be switched over to proper FITS files when happy
    # with format, etc.
    tmp = pyfits.open(hsdatafile)

    t = {}
    t['kmin']= tmp['K_MINMAX'].data['kmin']
    t['kmax']= tmp['K_MINMAX'].data['kmax']

    for i in range(12):
      t[i] = {}
      t[i]['a'] = tmp[i+2].data['a']
      t[i]['c'] = tmp[i+2].data['c']
    #self.data = numpy.load('hahnsavin.pkl', allow_pickle=True)

    self.data=t

  def get_coeffts(self, kappa, T):
    """
    Get the Maxwellian coefficients for the kapps distribution

    PARAMETERS
    ----------
    kappa : float
      kappa value for distribution. Must be > 1.5
    T : float
      temperature in K

    RETURNS
    -------
    Tlist : array(float)
      temperatures in K
    kappacoeff : array(float)
      the coefficients at each temperature in Tlist.
    """
    i = numpy.where((self.data['kmin'] < kappa) &\
                    (self.data['kmax'] >= kappa))[0][0]


    Tlist = self.data[i]['a']*T
    kappacoeff = numpy.zeros(len(Tlist))
    for ite in range(len(Tlist)):
      for ival in range(7):
        if numpy.isfinite(self.data[i]['c'][ite][ival]):
          kappacoeff[ite] += self.data[i]['c'][ite][ival]*kappa**(1.0*ival)

    return Tlist, kappacoeff

class KappaSession(spectrum.CIESession):
  """
  Load and generate a Kappa spectrum

  Parameters
  ----------
  linefile : string or HDUList, optional
    The line emissivity data file (either name or already open)
  cocofile : string or HDUList, optional
    The continuum emissivity data file (either name or already open)
  elements : arraylike(int), optional
    The atomic number of elements to include (default all)
  abundset : string
    The abundance set to use. Default AG89.

  Attributes
  ----------
  datacache : dict
    Any Atomdb FITS files which have to be opened are stored here
  spectra : KappaSpectra
    Object storing the actual spectral data
  elements : list(int)
    Nuclear charge of elements to include.
  default_abundset : string
    The abundance set used for the original emissivity file calculation
  abundset : string
    The abundance set to be used for the returned spectrum
  abundsetvector : array_like(float)
    The relative abundance between default_abundset and abundset for each element
  response_set : bool
    Have we loaded a response (or set a dummy response)
  dolines : bool
    Calculate line emission
  docont : bool
    Calculate continuum emission
  dopseudo : bool
    Calculate pseudocontinuum emission
  broaden_limit : float
    Apply broadening to lines with epsilon > this value (ph cm3 s-1)
  thermal_broadening : bool
    Apply thermal broadening to lines (default = False)
  velocity_broadening : float
    Apply velocity broadening with this velocity (km/s). If <=0, do not apply.

  Examples
  --------

  Create a session instance:

  >>> s=KappaSession()

  Set up the responses, in this case a dummy response from 0.1 to 10 keV

  >>> ebins = numpy.linspace(0.1,10,1000)
  >>> s.set_response(ebins, raw=True)

  Turn on thermal broadening

  >>> s.set_broadening(True)
  Will thermally broaden lines with emissivity > 1.000000e-18 ph cm3 s-1

  Return spectrum at 1.0keV with kappa = 3.1

  >>> spec = s.return_spectrum(1.0, 3.1)

  spec is in photons cm^3 s^-1 bin^-1; ebins are the bin edges (so spec is
  1 element shorter than ebins)
  """

  def __init__(self, linefile="$ATOMDB/apec_nei_line.fits",\
                     cocofile="$ATOMDB/apec_nei_comp.fits",\
                     kappadir = None,\
                     hsdatafile = "$ATOMDBKAPPA/hahnsavin.fits",\
                     ionrecdatafile = "$ATOMDBKAPPA/kappa_ir.fits",\
                     elements=[1,2,6,7,8,10,12,13,14,16,18,20,26,28],\
                      abundset='AG89'):
    """
    Initialization routine. Can set the line and continuum files here

    Input
    -----
    linefile : str or HDUList
      The filename of the line emissivity data, or the opened file.
    cocofile : str or HDUList
      The filename of the continuum emissivity data, or the opened file.
    elements : array_like(int)
      The atomic numbers of the elements to include. Defaults to all (1-30)
    abundset : string
      The abundance set to use. Defaults to AG89. See atomdb.set_abundance
      for list of options.
    """


    self.datacache={}

    # Open up the APEC files
    self.set_apec_files(linefile, cocofile)


    # if elements are specified, use them. Otherwise, use Z=1-30
    if util.keyword_check(elements):
      self.elements = elements
    else:
      self.elements=list(range(1,const.MAXZ_NEI+1))

    # define the directories & other data files
    if kappadir==None:
      # set to the directory where this file sits
      self.kappadir = os.path.dirname(os.path.realpath(__file__))
      os.environ['ATOMDBKAPPA'] = self.kappadir
    else:
      os.environ['ATOMDBKAPPA'] = kappadir

    self.hsdatafile = os.path.expandvars(hsdatafile)
    self.ionrecdatafile = os.path.expandvars(ionrecdatafile)

    # a hold for the spectra
    self.spectra=KappaSpectrum(self.linedata, self.cocodata, \
                               self.hsdatafile, self.ionrecdatafile,
                               elements = self.elements)


    # Set both the current and the default abundances to those that
    # the apec data was calculated on
    self.abundset=self.linedata[0].header['SABUND_SOURCE']
    self.default_abundset=self.linedata[0].header['SABUND_SOURCE']

    self.abundsetvector = numpy.zeros(const.MAXZ_NEI+1)
    for Z in self.elements:
      self.abundsetvector[Z] = 1.0

    #  but if another vector was already specified, use this instead
    if util.keyword_check(abundset):
      self.set_abundset(abundset)

    self.abund = numpy.zeros(const.MAXZ_NEI+1)

    for Z in self.elements:
      self.abund[Z]=1.0

    # Set a range of parameters which can be overwritten later
    self.response_set = False # have we loaded a response file?
    self.dolines=True # Include lines in spectrum
    self.docont=True # Include continuum in spectrum
    self.dopseudo=True # Include pseudo continuum in spectrum
    self.set_broadening(False, broaden_limit=1e-18)
    self.cdf = spectrum._Gaussian_CDF()

  def set_apec_files(self, linefile="$ATOMDB/apec_nei_line.fits",\
                     cocofile="$ATOMDB/apec_nei_comp.fits"):
    """
    Set the apec line and coco files, and load up their data

    Parameters
    ----------
    linefile : str or HDUList
      The filename of the line emissivity data, or the opened file.
    cocofile : str or HDUList
      The filename of the continuum emissivity data, or the opened file.

    Returns
    -------
    None

    Notes
    -----
    Updates self.linefile, self.linedata, self.cocofile and self.cocodata
    """
    if util.keyword_check(linefile):
      if isinstance(linefile, str):
        lfile = os.path.expandvars(linefile)
        if not os.path.isfile(lfile):
          print("*** ERROR: no such file %s. Exiting ***" %(lfile))
          return -1
        self.linedata = pyfits.open(lfile)
        self.linefile = lfile

      elif isinstance(linefile, pyfits.hdu.hdulist.HDUList):
        # no need to do anything, file is already open
        self.linedata=linefile
        self.linefile=linefile.filename()

      else:
        print("Unknown data type for linefile. Please pass a string or an HDUList")

    if util.keyword_check(cocofile):

      if isinstance(cocofile, str):

        cfile = os.path.expandvars(cocofile)
        if not os.path.isfile(cfile):
          print("*** ERROR: no such file %s. Exiting ***" %(cfile))
          return -1
        self.cocodata=pyfits.open(cfile)
        self.cocofile=cfile

      elif isinstance(cocofile, pyfits.hdu.hdulist.HDUList):
        # no need to do anything, file is already open
        self.cocodata=cocofile
        self.cocofile=cocofile.filename()

      else:
        print("Unknown data type for cocofile. Please pass a string or an HDUList")


  def return_linelist(self, Te, tau, specrange, specunit='A', \
                               teunit='keV', apply_aeff=False, develop=False):
    """
    Get the list of line emissivities vs wavelengths


    Parameters
    ----------
    Te : float
      Temperature in keV or K
    tau : float
      ionization timescale, ne * t (cm^-3 s).
    specrange : [float, float]
      Minimum and maximum values for interval in which to search
    specunit : {'Angstrom','keV'}
      Units for specrange
    teunit : {'keV' , 'K'}
      Units of te (kev or K, default keV)
    apply_aeff : bool
      If true, apply the effective area to the lines in the linelist to
      modify their intensities.

    Returns
    -------
    linelist : array(dtype)
      The list of lines with lambda (A), energy (keV), epsilon (ph cm3 s-1),\
      epsilon_aeff (ph cm5 s-1) ion (string) and upper & lower levels.

    """

    print("WARNING: THIS IS IN DEVELOPMENT AND DOESN'T WORK YET")

    if not develop:
      print('exiting as not functional')
      return

    kT = util.convert_temp(Te, teunit, 'keV')

    el_list = self.elements
    ab = {}
    for Z in el_list:
      ab[Z] = self.abund[Z]*self.abundsetvector[Z]



    s= self.spectra.return_linelist(kT, tau, specrange=specrange, teunit='keV',\
                                        specunit=specunit, elements=self.elements,\
                                        abundances = ab)

    # do the response thing
    #resp  = s.response()

    if apply_aeff == True:
      ibin = numpy.zeros(len(s), dtype=int)
      for i, ss in enumerate(s):
        e = const.HC_IN_KEV_A/ss['Lambda']
        ibin[i] = numpy.where(self.specbins<e)[0][-1]

      s["Epsilon_Err"] = s['Epsilon']*self.aeff[ibin]

    return(s)


  def return_line_emissivity(self, Telist, taulist, Z, z1, up, lo, \
                             specunit='A', teunit='keV', \
                             apply_aeff=False, apply_abund=True,\
                             log_interp = True, init_pop='ionizing'):
    """
    Get line emissivity as function of Te, tau. Assumes ionization from neutral.


    Parameters
    ----------
    Telist : float or array(float)
      Temperature(s) in keV or K
    taulist : float
      ionization timescale(s), ne * t (cm^-3 s).
    Z : int
      nuclear charge of element
    z1 : int
      ion charge +1 of ion
    up : int
      upper level for transition
    lo : int
      lower level for transition
    specunit : {'Angstrom','keV'}
      Units for wavelength or energy (a returned value)
    teunit : {'keV' , 'K'}
      Units of Telist (kev or K, default keV)
    apply_aeff : bool
      If true, apply the effective area to the line emissivity in the
      linelist to modify their intensities.
    apply_abund : bool
      If true, apply the abundance set in the session to the result.
    log_interp : bool
      Interpolate between temperature on a log-log scale (default).
      Otherwise linear

    Returns
    -------
    ret : dict
      Dictionary containing:
      Te, tau, teunit: as input
      wavelength : line wavelength (A)
      energy : line energy (keV)
      epsilon : emissivity in ph cm^3 s-1 (or ph cm^5 s^-1 if apply_aeff=True)
                first index is temperature, second is tau.

    """

    print(" NOTE NOT FUNCTIONAL YET - returning")
    return

    Tevec, Teisvec = util.make_vec(Telist)
    tauvec, tauisvec = util.make_vec(taulist)


    kTlist = util.convert_temp(Tevec, teunit, 'keV')
    if apply_abund:
      ab = self.abund[Z]*self.abundsetvector[Z]
    else:
      ab = 1.0

    eps = numpy.zeros([len(Tevec), len(tauvec)])
    ret={}
    ret['wavelength'] = None
    for itau, tau in enumerate(tauvec):
      for ikT, kT in enumerate(kTlist):
        e, lam = self.spectra.return_line_emissivity(kT, tau, Z, z1, \
                                                     up, lo, \
                                                     specunit='A', \
                                                     teunit='keV', \
                                                     abundance=ab,\
                                                     init_pop=init_pop)

        eps[ikT, itau] = e
        if lam != False:
          ret['wavelength'] = lam * 1.0
        else:
          ret['wavelength'] = None

    ret['Te'] = Telist
    ret['tau'] = taulist
    ret['teunit'] = teunit
    if ret['wavelength'] != None:
      ret['energy'] = const.HC_IN_KEV_A/ret['wavelength']
    else:
      ret['energy'] = None


    if apply_aeff == True:
      e = ret['energy']
      ibin = numpy.where(self.specbins<e)[0][-1]

      eps = eps*self.aeff[ibin]


    # now correct for vectors

    if not tauisvec:
      eps=eps[:,0]
      if not Teisvec:
        eps = eps[0]
    else:
      if not Teisvec:
        eps = eps[0,:]

    ret['epsilon'] = eps

    return ret

  def return_spectrum(self,  Te, kappa, teunit='keV', nearest=False,\
                      get_nearest_t=False, log_interp=True):
    """
    Get the spectrum at an exact temperature.
    Interpolates between 2 neighbouring spectra

    Finds HDU with kT closest to desired kT in given line or coco file.

    Opens the line or coco file, and looks for the header unit
    with temperature closest to te. Use result as index input to make_spectrum

    Parameters
    ----------
    Te : float
      Temperature in keV or K
    teunit : {'keV' , 'K'}
      Units of te (kev or K, default keV)
    raw : bool
      If set, return the spectrum without response applied. Default False.
    nearest : bool
      If set, return the spectrum from the nearest tabulated temperature
      in the file, without interpolation
    get_nearest_t : bool
      If set, and `nearest` set, return the nearest tabulated temperature
      as well as the spectrum.

    Returns
    -------
    spectrum : array(float)
      The spectrum in photons cm^5 s^-1 bin^-1, with the response, or
      photons cm^3 s^-1 bin^-1 if raw is set.
    nearest_T : float, optional
      If `nearest` is set, return the actual temperature this corresponds to.
      Units are same as `teunit`
    """

    # Check that there is a response set
    if not self.response_set:
      raise util.ReadyError("Response not yet set: use set_response to set.")

    el_list = self.elements
    ab = {}
    for Z in el_list:
      ab[Z] = self.abund[Z]*self.abundsetvector[Z]

    self.spectra.ebins = self.specbins
    self.spectra.ebins_checksum=hashlib.md5(self.spectra.ebins).hexdigest()
    s= self.spectra.return_spectrum(Te, kappa, teunit=teunit, elements = el_list, \
                                    abundances=ab, log_interp=True,\
                                    broaden_object=self.cdf)

    ss = self._apply_response(s)

    return ss

























class KappaSpectrum(spectrum._CIESpectrum):
  """
  A class holding the emissivity data for NEI emission, and returning
  spectra

  Parameters
  ----------
  linefile : string or HDUList, optional
    The line emissivity data file (either name or already open)
  cocofile : string or HDUList, optional
    The continuum emissivity data file (either name or already open)
  elements : arraylike(int), optional
    The atomic number of elements to include (default all)
  abundset : string
    The abundance set to use. Default AG89.

  Attributes
  ----------
  session : CIESession
    The parent CIESession
  SessionType : string
    "CIE"
  spectra : dict of ElementSpectra
    a dictionary containing the emissivity data for each HDU,
    subdivided by element (spectra[12][18] is an ElementSpectrum object
    containing the argon data for the 12th HDU)
  kTlist : array
    The temperatures for each emissivity HDU, in keV
  logkTlist : array
    log of kTlist
  """

  def __init__(self, linedata, cocodata, hsdatafile, ionrecdatafile, elements):
    """
    Initializes the code. Populates the line and emissivity data in all
    temperature HDUs.

    Parameters
    ----------
    linedata :
      The parent CIESession
    """
    self.elements=elements
    self.datacache={}
    self.SessionType = 'NEI'

    picklefname = os.path.expandvars('$ATOMDB/spectra_%s_%s.pkl'%\
                                (linedata[0].header['CHECKSUM'],\
                                 cocodata[0].header['CHECKSUM']))


    if os.path.isfile(picklefname):
      self.spectra = pickle.load(open(picklefname,'rb'))
      self.kTlist = self.spectra['kTlist']
    else:
      self.spectra={}
      self.kTlist = numpy.array(linedata[1].data['kT'].data)
      self.spectra['kTlist'] = numpy.array(linedata[1].data['kT'].data)
      for ihdu in range(len(self.kTlist)):
        self.spectra[ihdu]={}
        self.spectra[ihdu]['kT'] = self.kTlist[ihdu]
        ldat = numpy.array(linedata[ihdu+2].data.data)
        cdat = numpy.array(cocodata[ihdu+2].data.data)


        Zarr = numpy.zeros([len(ldat), const.MAXZ_NEI+1], dtype=bool)
        Zarr[numpy.arange(len(ldat), dtype=int), ldat['Element']]=True


        for Z in range(1,const.MAXZ_NEI+1):

          if not Z in self.spectra[ihdu].keys():
            self.spectra[ihdu][Z] = {}

          for z1 in range(1,Z+2):
            isz1 = (ldat['Ion_drv']==z1)
            isgood = isz1*Zarr[:,Z]
            ccdat = cdat[(cdat['Z']==Z) & (cdat['rmJ']==z1)]

            if len(ccdat)==0:
              ccdat = [False]
            self.spectra[ihdu][Z][z1]=spectrum._ElementSpectrum(ldat[isgood],\
                                                  ccdat[0], Z, z1_drv=z1)


      pickle.dump(self.spectra, open(picklefname,'wb'))
    self.logkTlist=numpy.log(self.kTlist)

    # now repeat for hahn savin data
    self.hsdata = hs_data(hsdatafile)

    # now repeat for ionization and recombination
    self.ionrecdata = ir_data(irfile = ionrecdatafile, elements=self.elements)



  def calc_ionrec_rate(self, tkappa, ckappa, elements):
    """
    Calculate the ionization and recombination rates for a kappa
    distribution, by summing maxwellians

    PARAMETERS
    ----------
    tkappa : array(float)
      temperatures of each maxwellian component (K)
    ckappa : array(float)
      norm of each maxwellian component
    elements : array(int)
      atomic number of elements to calculate rates for

    RETURNS
    -------
    ionrate : dict
      e.g. ionrate[16] is the ionization rate coefft for sulphur 1 through 17, in cm^3 s-1
    recrate : dict
      e.g. recrate[16] is the recombiation rate coefft for sulphur 1 through 17, in cm^3 s-1
    """

    ircoeffts = self.ionrecdata.get_ir_rate(tkappa)
    ionrate = {}
    recrate = {}

    for Z in elements:
      ionrate[Z] = numpy.zeros(Z)
      recrate[Z] = numpy.zeros(Z)
      for z1 in range(1,Z+1):
        ionrate[Z][z1-1]=sum( ircoeffts[Z]['ion'][z1-1,:]*ckappa)
        recrate[Z][z1-1]=sum( ircoeffts[Z]['rec'][z1-1,:]*ckappa)
    return ionrate, recrate



  def return_oneT_spectrum(self, Te, Z, z1, epslimit, teunit='keV', log_interp=True,\
                           broaden_object=False, ikT=False, f=False):
    """
    return a single element, single ion, spectrum, interpolating
    appropriately between neighboring temperature bins

    """
    T = util.convert_temp(Te, teunit, 'K')
    kT = util.convert_temp(Te, teunit, 'keV')


    # Recalc fractions if required
    if (type(ikT)==bool) | (type(f)==bool):
      ikT, f = self.get_nearest_Tindex(kT, teunit='keV',  log_interp=log_interp)


    # ok, get the spectra
    stot=0.0
    for i in range(len(ikT)):

      # get the spectrum
      sss = self.spectra[ikT[i]][Z][z1].return_spectrum(self.ebins,\
                                   kT,\
                                   ebins_checksum = self.ebins_checksum,\
                                   thermal_broadening = self.thermal_broadening,\
                                   broaden_limit = epslimit,\
                                   velocity_broadening = self.velocity_broadening,\
                                   broaden_object=broaden_object)
      # add it appropriately
      if log_interp:
        stot += numpy.log(sss+const.MINEPSOFFSET)*f[i]
      else:
        stot +=sss*f[i]
    # now handle the sum

    stot = numpy.exp(stot)-const.MINEPSOFFSET*len(f)
    stot[stot<0] = 0.0
    return stot




  def return_spectrum(self, Te, kappa, teunit='keV',
                             elements=False, \
                             abundances=False, log_interp=True, broaden_object=False):

    """
    Return the spectrum of the element on the energy bins in
    self.session.specbins

    Parameters
    ----------
    Te : float
      Electron temperature (default, keV)
    tau : float
      ionization timescale, ne * t (cm^-3 s).
    teunit : string
      Units of kT (keV by default, K also allowed)
    nearest : bool
      If True, return spectrum for the nearest temperature index.
      If False, use the weighted average of the (log of) the 2 nearest indexes.
      default is False.

    Returns
    -------
    spec : array(float)
      The element's emissivity spectrum, in photons cm^3 s^-1 bin^-1
    """

    # get kT in keV

    T = util.convert_temp(Te, teunit, 'K')
    kT = util.convert_temp(Te, teunit, 'keV')


    # find the correct coefficients here
    tkappa_all, ckappa_all = self.hsdata.get_coeffts(kappa, T)

    # filter out of range ones
    ckappa = ckappa_all[(tkappa_all >= 1e4) & (tkappa_all <= 1e9)]
    if len(ckappa) < len(tkappa_all):
      print("Note: only using %i of %i requested Maxwellian components as they are inside the 10^4 to 10^9K range"%(len(ckappa), len(tkappa_all)))

    tkappa = tkappa_all[(tkappa_all >= 1e4) & (tkappa_all <= 1e9)]

    # check the params:
    if elements==False:
      elements = self.elements
#      elements=range(1,const.MAXZ_NEI+1)


    if abundances == False:
      abundances = {}
      for Z in elements:
        abundances[Z] = 1.0

    s = 0.0
    #ionrate, recrate = self.calc_ionrec_rate(tkappa_all, ckappa_all, elements)
    self.calc_ionbal(tkappa_all, ckappa_all, elements)

    for Z in elements:
      abund = abundances[Z]
      if abund > 0:

        # solve the ionization balance
     #   self.ionbal[Z] = apec.solve_ionbal(ionrate[Z], recrate[Z])
        s1 = 0.0
        ionfrac = self.ionbal[Z]

        for ik, tk in enumerate(tkappa):
          ikT, f = self.get_nearest_Tindex(tk, teunit='K',  log_interp=True)
          s1=0.0
          for z1 in range(1, Z+2):
            if ionfrac[z1-1]>1e-10:

              # calculate minimum emissivitiy to broaden, accounting for ion
              # and element abundance.
              epslimit =  self.broaden_limit/(abund*ionfrac[z1-1])

              sss  = self.return_oneT_spectrum(kT, Z, z1, epslimit, teunit='keV', log_interp=log_interp,\
                                       broaden_object=broaden_object, ikT=ikT, f=f)

              sss*=abund*ionfrac[z1-1]

              s1+=sss

          s+=s1*ckappa[ik]
    return s

  def calc_ionbal(self, tkappa_all, ckappa_all, elements):
    ionrate, recrate = self.calc_ionrec_rate(tkappa_all, ckappa_all, elements)

    self.ionbal={}
    for Z in elements:
      # solve the ionization balance
      self.ionbal[Z] = apec.solve_ionbal(ionrate[Z], recrate[Z])

  def return_line_emissivity(self, Te, tau, Z, z1, up, lo, specunit='A',
                             teunit='keV', abundance=1.0,
                             log_interp = True, init_pop = 'ionizing'):
    """
    Return the emissivity of a line at kT, tau. Assumes ionization from neutral for now


    Parameters
    ----------
    Te : float
      Temperature in keV or K
    tau : float
      ionization timescale, ne * t (cm^-3 s).
    Z : int
      nuclear charge of element
    z1 : int
      ion charge +1 of ion
    up : int
      upper level for transition
    lo : int
      lower level for transition
    specunit : {'Angstrom','keV'}
      Units for wavelength or energy (a returned value)
    teunit : {'keV' , 'K'}
      Units of Telist (kev or K, default keV)
    abundance : float
      Abundance to multiply the emissivity by

    init_pop : string or float
      If string:
        if 'ionizing' : all ionizing from neutral (so [1,0,0,0...])
        if 'recombining': all recombining from ionized (so[...0,0,1])
        if array of length (Z+1) : the acutal fractional populations
        if single float : the temperature (same units as Te)

    Returns
    -------
    Emissivity : float
      Emissivity in photons cm^3 s^-1
    spec : float
      Wavelength or Energy of line, depending on specunit
    """

    import collections

    kT = util.convert_temp(Te, teunit, 'keV')

    ikT, f = self.get_nearest_Tindex(kT, \
                                     teunit='keV', \
                                     nearest=False, \
                                     log_interp=log_interp)
    #ikT has the 2 nearest temperature indexes
    # f has the fraction for each

    if type(init_pop) == str:
      if init_pop == 'ionizing':
        # everything neutral
        ipop = numpy.zeros(Z+1)
        ipop[0] = 1.0
      elif init_pop == 'recombining':
        # everything ionizing
        ipop = numpy.zeros(Z+1)
        ipop[-1] = 1.0
    else:
      if isinstance(init_pop, (collections.Sequence, numpy.ndarray)):
        if len(init_pop)==Z+1:
          ipop = init_pop
        else:
          pass
      else:
        kT_in = util.convert_temp(init_pop, teunit, 'keV')
        ipop = apec.solve_ionbal_eigen(Z, \
                                      kT_in, \
                                      teunit='keV', \
                                      datacache=self.datacache)


    ionfrac = apec.solve_ionbal_eigen(Z, \
                                      kT, \
                                      init_pop=ipop, \
                                      tau=tau, \
                                      teunit='keV', \
                                      datacache=self.datacache)

    eps = 0.0
    lam = 0.0


      # find lines which match
    for z1_drv in range(1,Z+2):
      # ions which don't exist get skipped
      if ionfrac[z1_drv-1] <= 1e-10: continue
      eps_in = numpy.zeros(len(ikT))

      for i in range(len(ikT)):
        iikT =ikT[i]

        llist = self.spectra[iikT][Z][z1_drv].return_linematch(Z,z1,up,lo)

        for line in llist:
          # add emissivity
          eps_in[i] += line['Epsilon']
          lam = line['Lambda']

      if log_interp:
        eps_out = 0.0
        for i in range(len(ikT)):
          eps_out += f[i]*numpy.log(eps_in[i]+const.MINEPSOFFSET)
        eps += numpy.exp(eps_out-const.MINEPSOFFSET)*abundance * ionfrac[z1_drv-1]
      else:
        eps_out = 0.0
        for i in range(len(ikT)):
          eps_out += f[i]*eps_in[i]
        eps += eps_out*abundance * ionfrac[z1_drv-1]


    if specunit == 'keV':
      lam = const.HC_IN_KEV_A/lam
    return eps, lam

  def return_linelist(self,  Te, tau, Te_init=False,
                      teunit='keV', nearest = False, specrange=False,
                      specunit='A', elements=False, abundances=False,\
                      init_pop = 'ionizing', log_interp=True):

    """
    Return the linelist of the element

    Parameters
    ----------

    Te : float
      Electron temperature (default, keV)
    teunit : string
      Units of kT (keV by default, K also allowed)
    nearest : bool
      If True, return spectrum for the nearest temperature index.
      If False, use the weighted average of the (log of) the 2 nearest indexes.
      default is False.
    specrange : [float, float]
      Minimum and maximum values for interval in which to search
    specunit : {'Ansgtrom','keV'}
      Units for specrange (default A)


    """
    # get kT in keV
    kT = util.convert_temp(Te, teunit, 'keV')


    ikT, f = self.get_nearest_Tindex(kT, teunit='keV', nearest=nearest)

    if abundances == False:
      abundances = {}
      for Z in elements:
        abundances[Z] = 1.0

    linelist = False


    # Cycle through each element
    for Z in elements:


      abund = abundances[Z]

      # Skip if abundance is low
      if abund > 0:
        elemlinelist = {}


        # Get initial ion population for element
        if type(init_pop) == str:
          if init_pop == 'ionizing':
            # everything neutral
            ipop = numpy.zeros(Z+1)
            ipop[0] = 1.0
          elif init_pop == 'recombining':
            # everything ionizing
            ipop = numpy.zeros(Z+1)
            ipop[-1] = 1.0
        else:
          if isinstance(init_pop, (collections.Sequence, numpy.ndarray)):
            if len(init_pop)==Z+1:
              ipop = init_pop
            else:
              pass
          else:
            kT_in = util.convert_temp(init_pop, teunit, 'keV')
            ipop = apec.solve_ionbal_eigen(Z, \
                                          kT_in, \
                                          teunit='keV', \
                                          datacache=self.datacache)




        # calculate final ion population for element
        ionfrac = apec.solve_ionbal_eigen(Z, \
                                          kT, \
                                          init_pop=ipop, \
                                          tau=tau, \
                                          teunit='keV', \
                                          datacache=self.datacache)

        # go through the 2 nearest emissivity temperatures
        for i in range(len(ikT)):
          iikT = ikT[i]
          elemlinelist[iikT] = False

          # go ion by ion
          for z1_drv in range(1, Z+2):

            # skip if ion fraction is low
            if ionfrac[z1_drv-1] < 1e-10: continue

            # list all the lines for the ion_drv
            ss = self.spectra[ikT[i]][Z][z1_drv].return_linelist(specrange,\
                                    teunit='keV', specunit=specunit)

            # if 1 or more lines found, do something
            if len(ss) > 0:

              # adjust line emissivty by abundance and ion fraction
              ss['Epsilon']*=abund*ionfrac[z1_drv-1]

              # if this is the first set of lines, add them
              if elemlinelist[iikT]==False:
                elemlinelist[iikT] = ss
              else:
              # otherwise, merge them in. No separation by driving ion.
                isnew = numpy.zeros(len(ss), dtype=bool)

                for inew, new in enumerate(ss):
                  imatch = numpy.where((new['Element']==elemlinelist[iikT]['Element']) &\
                                       (new['Ion']==elemlinelist[iikT]['Ion']) &\
                                       (new['UpperLev']==elemlinelist[iikT]['UpperLev']) &\
                                       (new['LowerLev']==elemlinelist[iikT]['LowerLev']))[0]
                  if len(imatch)==1:
                    # if the same line already exists, add to its flux
                    elemlinelist[iikT][imatch[0]]['Epsilon']+=new['Epsilon']
                  else:
                    # otherwise, declare it as a new line, ready to append
                    isnew[inew]=True

                s = sum(isnew)
                if s > 0:
                  # append any new lines to the end of elemlinelist
                  elemlinelist[iikT] = numpy.append(elemlinelist[iikT], ss[isnew])

          # At this point, elemlinelist[ikT] contains the flux in each line for a single ikT.
          # We need to adjust these, and then ultimately merge them with the next ikT with
          # a suitbale multiplicative factor.

          # OK, so now multiply all the line emissivities by the appropriate factor
          if elemlinelist[iikT] == False: continue

          if log_interp:
            elemlinelist[iikT]['Epsilon'] = f[i] * numpy.log(elemlinelist[iikT]['Epsilon']+ const.MINEPSOFFSET)

          else:
            elemlinelist[iikT]['Epsilon'] = f[i] * elemlinelist[iikT]['Epsilon']

        # now merge these 2 temperature results
        ikTlist = ikT*1
        ikTkeep = numpy.ones(len(ikTlist), dtype=bool)
        for i,iikT in enumerate(ikT):
          if elemlinelist[iikT]==False:
            ikTkeep[i] = False

        if sum(ikTkeep)==0: continue

        if sum(ikTkeep)==1:
          ikTlist = ikTlist[ikTkeep]

        if len(ikTlist) > 1:
          # find matches for each line
          iikT = ikTlist[1]
          hasmatch = numpy.zeros(len(elemlinelist[iikT]), dtype=bool)

          for iline, line in enumerate(elemlinelist[iikT]):

            imatch = numpy.where((elemlinelist[ikTlist[0]]['Element']==line['Element']) &\
                                 (elemlinelist[ikTlist[0]]['Ion']==line['Ion']) &\
                                 (elemlinelist[ikTlist[0]]['UpperLev']==line['UpperLev']) &\
                                 (elemlinelist[ikTlist[0]]['LowerLev']==line['LowerLev']))[0]
            if len(imatch)==0:
              line['Epsilon'] = numpy.exp(line['Epsilon']-const.MINEPSOFFSET)
            else:
              hasmatch[iline] = True
              itmp = imatch[0]
              elemlinelist[ikTlist[0]][itmp]['Epsilon'] += line['Epsilon']

          if log_interp:
            elemlinelist[ikTlist[0]]['Epsilon'] = numpy.exp(elemlinelist[ikTlist[0]]['Epsilon']- const.MINEPSOFFSET)

          # now append the unmatched
          elemlinelist = numpy.append(elemlinelist[ikTlist[0]], elemlinelist[ikTlist[1]][~hasmatch])

        else:
          if log_interp:
            elemlinelist = numpy.exp(elemlinelist[ikTlist[0]]['Epsilon']- const.MINEPSOFFSET)


        if linelist == False:
          linelist = numpy.zeros(0, dtype=elemlinelist.dtype)

        linelist=numpy.append(linelist, elemlinelist)


    return linelist





















MIN_IONBAL = 1e-10

def make_vec(d):
  """
  Create vector version of d, return True or false depending on whether
  input was vector or not

  Parameters
  ----------
  d: any scalar or vector
    The input

  Returns
  -------
  vecd : array of floats
    d as a vector (same as input if already an iterable type)
  isvec : bool
    True if d was a vector, otherwise False.


  """
  isvec=True
  try:
    _ = (e for e in d)
    vecd=d
  except TypeError:
    isvec=False
    vecd= numpy.array([d])

  return vecd,isvec


# some refence constants
INTERP_I_UPSILON=900     #/*!< Include both left & right boundaries */
UPSILON_COLL_COEFF = 8.629e-6 # /*!< sqrt{2 pi / kB} hbar^2/m_e^{1.5} */
INTERP_IONREC_RATE_COEFF =300
MAX_CHI = 200.0


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


class ir_data():
#-------------------------------------------------------------------------------

  def __init__(self, elements=False,\
               irfile = 'kappa_ir.fits'):

    # load up the kappa coefficients
    self.elements=elements
    ionrec_data = {}
    ir = pyfits.open(irfile)
    for irdat in ir['KAPPA_IR'].data:
      Z = irdat['ELEMENT']

      z1 = irdat['ION_INIT']
      if not Z in ionrec_data.keys():
        ionrec_data[Z] = {}
      if not z1 in ionrec_data[Z].keys():
        ionrec_data[Z][z1] = {}
        ionrec_data[Z][z1]['EA']=False
        ionrec_data[Z][z1]['CI']=False
        ionrec_data[Z][z1]['RR']=False
        ionrec_data[Z][z1]['DR']=False
      ionrec_data[Z][z1][irdat['TR_TYPE']]=irdat

    for ionpot in ir['IONPOT'].data:
      ionrec_data[ionpot['ELEMENT']][ionpot['ION']]['IONPOT'] = ionpot['IONPOT']

      ionrec_data[ionpot['ELEMENT']][ionpot['ION']]['IP_DERE'] = ionpot['IP_DERE']

    self.ionrecdata=ionrec_data

#-------------------------------------------------------------------------------
  def get_ir_rate(self, T):
    """
    Return the (Maxwellian) ionization and recombination rates at temperature T (in K)

    PARAMETERS
    ----------
    T : float or array(float)
      Temperature (Kelvin)
    """


    rates={}
    Tvec , wasTvec = make_vec(T)

    for Z in self.elements:
      rates[Z] ={}
      rates[Z]['ion'] = numpy.zeros([Z, len(Tvec)])
      rates[Z]['rec'] = numpy.zeros([Z, len(Tvec)])

      for z1 in range(1, Z+1):
        if self.ionrecdata[Z][z1]['CI'] != False:
          cirate = self.calc_ci_rate(Z, z1, Tvec)
        else:
          cirate = numpy.zeros(len(Tvec))

        if self.ionrecdata[Z][z1+1]['DR'] != False:
          drrate = self.calc_dr_rate(Z, z1+1, Tvec)
        else:
          drrate = numpy.zeros(len(Tvec))

        if self.ionrecdata[Z][z1+1]['RR'] != False:
          rrrate = self.calc_rr_rate(Z, z1+1, Tvec)
        else:
          rrrate = numpy.zeros(len(Tvec))

        if self.ionrecdata[Z][z1]['EA'] != False:
          earate = self.calc_ea_rate(Z, z1, Tvec)
        else:
          earate = numpy.zeros(len(Tvec))

        rates[Z]['ion'][z1-1,:] += cirate+earate
        rates[Z]['rec'][z1-1,:] += drrate+rrrate

    return rates

#-------------------------------------------------------------------------------

  def calc_maxwell_rate(par_type,\
                              min_temp,\
                              max_temp,\
                              temperatures,\
                              ionrec_par,\
                              ionpot, T, Z, degl, degu,\
                              force_extrap = True):
    """
    Interpolate ionization and recombination rates

    PARAMETERS
    ----------
    par_type : int
      number denoting parameter type. Should be >900 for CI
    min_temp : float
      minimum temperature in tabulation (K)
    max_temp : float
      maximum temperature in tabulation (K)
    temperatures: array(float)
      temperatures ionrec_par is tabulated on (K)
    ionrec_par: array(float)
      ionization & recombination parameters
    ionpot : float
      ionization potential (keV)
    T : float or array(float)
      temperatures to caculate the rates on (K)
    Z : int
      nuclear charge
    degl : int
      degeneracy of initial ion ground state
    degu : int
      degeneracy of final ion ground state
    force_extrap : bool
      Whether to extrappolate outside of the listed table

    RETURNS
    -------
    ir : float or array(float)
      The ionization or recombination rate coefficient, in cm^3 s^-1
    """



    Tvec, wasTvec = make_vec(T)

    # number of useful data points
    if (par_type > INTERP_IONREC_RATE_COEFF) &\
       (par_type <= INTERP_IONREC_RATE_COEFF+20):

      N_intep = par_type-INTERP_IONREC_RATE_COEFF

      # Check if input temperature was a vector. Convert to vector

      # extract the data, and convert to doubles to aviod numerical issues
      te_in = numpy.double(temperatures[:N_interp])
      ci_in = numpy.double(ionrec_par[:N_interp])

      # interpolate on log-log grid. Add 1e-30 to avoid interpolating log(0)
      ir = numpy.exp(interpolate.interp1d(numpy.log(te_in), \
                               numpy.log(ci_in+1e-30), \
                               kind=1, bounds_error=False,\
                               fill_value=numpy.nan)(numpy.log(Tvec)))-1e-30

      # let's deal with the near misses. This is necessary because sometimes
      # a value is deemed out of bounds even though it is only </> the min/max
      # due to floating point rounding issues. In these cases, set to the nearest

      nantest =  numpy.isnan(ir)


      for i in range(len(nantest)):
        if nantest[i]:
          if ((Tvec[i] > 0.99*min_temp)&\
              (Tvec[i] <= min_temp)):
            ir[i] = ionrec_par[0]


          if ((Tvec[i] < 1.01*max_temp)&\
              (Tvec[i] >= max_temp)):
            ir[i] = ci_in[-1]

      # now look for further out of bounds issues, if extrappolation is desired
      if force_extrap:

        nantest =  numpy.isnan(ir)
        for i in range(len(nantest)):
          # high T points are extrapolated as T**-(3/2)
          if (Tvec[i] > max_temp):
            ir[i]= si_in[-1] * (te_in[-1]/Tvec[i])**1.5

      # low T points are set to zero
      ir[numpy.isnan(ir)] = 0.0
      if wasTvec== False:
        ir=ir[0]
      return ir

    elif (par_type > INTERP_I_UPSILON) &\
       (par_type <= INTERP_I_UPSILON+20):

      N_interp = par_type - INTERP_I_UPSILON
      te_in = temperatures[:N_interp]
      ci_in = ionrec_par[:N_interp]
      chi  = dE / (const.KBOLTZ*Tvec)


      #it = numpy.where((Tvec>=min(te_in)) &\
      #               (Tvec<=max(ti_in)))[0]

      # fudge for dubious ionrec_par...
      ci_in[ci_in < 0.0] = 0.0


      #if len(it) > 0:
      upsilon = interpolate.interp1d(numpy.log(te_in), \
                                     numpy.log(ci_in+1e-30), \
                                     bounds_error=False, \
                                     fill_value=numpy.nan)(numpy.log(Tvec))

      upsilon = numpy.exp(upsilon)-1e-30
      calc_type = const.EI_UPSILON


      upsilon[upsilon < 0] = 0.0

    # this is the same as the electron-impact version, but with extra factor of pi
      ir = UPSILON_COLL_COEFF * upsilon * \
                 numpy.exp(-chi) / (numpy.pi* numpy.sqrt(Tvec)*degl)

      ir[chi < MAX_CHI] = 0.0


      # low T points are set to zero
      ir[numpy.isnan(ir)] = 0.0

      if wasTvec== False:
        ir=ir[0]
      return ir

#-------------------------------------------------------------------------------

  def calc_ci_rate(self, Z, z1, T):
    """
    Calculate the collisional ionization rate

    PARAMETERS
    ----------
    Z : int
      element nuclear charge
    z1 : int
      ion charge +1. This is for the lowest charge state in the ionization
      or recombination process (so 3 for O III-> O IV and O IV -> O III)
    T : float or array(floats)
      Temperature (Kelvin) to calculate rates at

    RETURNS
    -------
    ci : float or array(float)
      The collisional ionization rate coefficient in cm^3 s^-1
    """

    cidat = self.ionrecdata[Z][z1]['CI']

    if ((cidat['par_type']>const.INTERP_I_UPSILON) & \
        (cidat['par_type']<=const.INTERP_I_UPSILON+20)):


      ionpot = self.ionrecdata[Z][z1]['IONPOT']

      #  THIS IS A HACK. FIXME  #

      degl = 1
      degu = 1

      # END HACK #

      ci = calc_maxwell_rate(cidat['PAR_TYPE'],\
                         cidat['MIN_TEMP'],\
                         cidat['MAX_TEMP'],\
                         cidat['TEMPERATURE'],\
                         cidat['IONREC_PAR'],\
                         ionpot/1e3, T, Z, degl, degu)


    elif ((cidat['par_type']>const.INTERP_I_UPSILON) & \
        (cidat['par_type']<=const.INTERP_I_UPSILON+20)):
      ci = calc_maxwell_rate(cidat['PAR_TYPE'],\
                         cidat['MIN_TEMP'],\
                         cidat['MAX_TEMP'],\
                                 cidat['TEMPERATURE'],\
                                 cidat['IONREC_PAR'],\
                                 ionpot/1e3, T, Z, degl, degu)


    elif ((cidat['par_type']>const.CI_DERE) &\
          (cidat['par_type']<=const.CI_DERE+20)):
      ionpot = self.ionrecdata[Z][z1]['IP_DERE']
      Tvec, wasTvec = make_vec(T)
      ci = atomdb._calc_ionrec_ci(cidat, Tvec, extrap=True, ionpot=ionpot)

    else:
      Tvec, wasTvec = make_vec(T)
      ci = atomdb._calc_ionrec_ci(cidat,Tvec, extrap=True)


    return ci


#-------------------------------------------------------------------------------

  def calc_dr_rate(self, Z, z1, T):
    """
    Calculate the dielectronic recombination rate

    PARAMETERS
    ----------
    Z : int
      element nuclear charge
    z1 : int
      ion charge +1. This is for the lowest charge state in the ionization
      or recombination process (so 3 for O III-> O IV and O IV -> O III)
    T : float or array(floats)
      Temperature (Kelvin) to calculate rates at

    RETURNS
    -------
    dr : float or array(float)
      The dielectronic recombinatino rate coefficient in cm^3 s^-1
    """

    drdat = self.ionrecdata[Z][z1]['DR']

    dr = atomdb._calc_ionrec_dr(drdat, T, extrap=True)
    return dr
#-------------------------------------------------------------------------------

  def calc_rr_rate(self, Z, z1, T):
    """
    Calculate the dielectronic recombination rate

    PARAMETERS
    ----------
    Z : int
      element nuclear charge
    z1 : int
      ion charge +1. This is for the lowest charge state in the ionization
      or recombination process (so 3 for O III-> O IV and O IV -> O III)
    T : float or array(floats)
      Temperature (Kelvin) to calculate rates at

    RETURNS
    -------
    rr : float or array(float)
      The dielectronic recombinatino rate coefficient in cm^3 s^-1
    """

    rrdat = self.ionrecdata[Z][z1]['RR']

    rr = atomdb._calc_ionrec_rr(rrdat, T, extrap=True)
    return rr
#-------------------------------------------------------------------------------

  def calc_ea_rate(self, Z, z1, T):
    """
    Calculate the dielectronic recombination rate

    PARAMETERS
    ----------
    Z : int
      element nuclear charge
    z1 : int
      ion charge +1. This is for the lowest charge state in the ionization
      or recombination process (so 3 for O III-> O IV and O IV -> O III)
    T : float or aeaay(floats)
      Temperature (Kelvin) to calculate rates at

    RETURNS
    -------
    ea : float or aeaay(float)
      The dielectronic recombinatino rate coefficient in cm^3 s^-1
    """

    eadat = self.ionrecdata[Z][z1]['EA']

    ea = atomdb._calc_ionrec_ea(eadat, T, extrap=True)
    return ea


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



# class kappamodel():
  # """
  # Class for all things kappa
  # """

  # def __init__(self, elements=[1,2,6,7,8,10,12,13,14,16,18,20,26,28],\
               # linefile = "$ATOMDB/apec_nei_line.fits",\
               # compfile = "$ATOMDB/apec_nei_comp.fits"):

    # # load up the kappa coefficients
    # self.hs_data = hs_data()
    # self.elements = elements
    # self.abund = {}
    # for Z in self.elements:
      # self.abund[Z] = 1.0
    # self.ionrec_data = irdata(elements = self.elements)

    # # for spectra
    # self.ebins = False
    # self.ebins_checksum = False
    # self.spectra = {}

    # self.linedata = pyatomdb.pyfits.open(os.path.expandvars(linefile))
    # self.compdata = pyatomdb.pyfits.open(os.path.expandvars(compfile))


    # self.spectra['temperatures'] = self.linedata[1].data['kT']/const.KBOLTZ
    # for Z in self.elements:
      # self.spectra[Z] = {}
      # #for z1 in range(1, Z+2):
      # #  self.spectra[Z][z1] = IonSpectrum(Z, z1, self.spectra['temperatures'], self.linedata, self.compdata)
    # print("Kappa model ready for use. Note that the first time a spectrum "+\
          # "is calculated it can take up to a  minute to finish. It gets much "+\
          # "faster after that, I promise.")


  # def get_kappa_coeffts(self, kappa, T):
    # """
    # Return the kappa coefficients and temperatures at desired temperature

    # PARAMETERS
    # ----------
    # kappa : float
      # kappa coefficient (>1.5)
    # T : float
      # temperature (K)

    # RETURNS
    # -------
    # Tkappa : array(float)
      # temperatures required for kappa calculation (K)
    # ckappa : array(float)
      # coefficients at these temperatures
    # """
    # Tkappa, ckappa = self.hs_data.get_coeffts(kappa,T)

    # return(Tkappa, ckappa)

  # def calc_ionrec_rate(self, kappa, T):
    # """
    # Calculate the ionization and recombination rates for a kappa
    # distribution, by summing maxwellians

    # PARAMETERS
    # ----------
    # kappa : float
      # kappa coefficient (>1.5)
    # T : float
      # temperature (K)

    # RETURNS
    # -------
    # ionrate : dict
      # e.g. ionrate[16] is the ionization rate coefft for sulphur 1 through 17, in cm^3 s-1
    # recrate : dict
      # e.g. recrate[16] is the recombiation rate coefft for sulphur 1 through 17, in cm^3 s-1
    # """

    # tkappa, ckappa = self.get_kappa_coeffts( kappa, T)
    # # filter OOB kappa:
# #    ckappa = ckappa[ (tkappa >=1e4) & (tkappa <= 1e9)]
# #    tkappa = tkappa[ (tkappa >=1e4) & (tkappa <= 1e9)]

    # ircoeffts = self.ionrec_data.get_ir_rate(tkappa)


    # ionrate = {}
    # recrate = {}


    # for Z in self.elements:
      # ionrate[Z] = numpy.zeros(Z)
      # recrate[Z] = numpy.zeros(Z)
      # for z1 in range(1,Z+1):
        # ionrate[Z][z1-1]=sum( ircoeffts[Z]['ion'][z1-1,:]*ckappa)
        # recrate[Z][z1-1]=sum( ircoeffts[Z]['rec'][z1-1,:]*ckappa)

    # self.tkappa = tkappa
    # self.ckappa = ckappa
    # return ionrate, recrate


  # def calc_ionbal(self, kappa, T, elements):
    # """
    # Calculate the ionization balance for a kappa distribution by summing
    # Maxwellians

    # PARAMETERS
    # ----------
    # kappa : float
      # kappa coefficient (>1.5)
    # T : float
      # temperature (K)

    # RETURNS
    # -------
    # ionbal : dict
      # e.g. ionbal[16] is a 17 element array with the fractional abundance of each ion of S, starting with neutral
    # """

    # ionrate, recrate = self.calc_ionrec_rate(kappa, T, elements)


    # ionbal = {}
    # for Z in elements:
      # ionbal[Z] = pyatomdb.apec.solve_ionbal(ionrate[Z], recrate[Z])
    # self.ionbal = ionbal


    # return ionbal


  # def set_ebins(self, ebins, ebins_checksum = False):
    # """
    # Set the energy bins for the spectrum being returned.

    # PARAMETERS
    # ----------
    # ebins : array(float)
      # Energy bin edges (keV)
    # ebins_checksum : string, optional
      # The hex digest of the md5 sum of ebins. Used to check for changes.
    # """

    # if ebins_checksum == False:
      # ebins_checksum = hashlib.md5(ebins).hexdigest()

    # self.ebins = ebins

    # if ebins_checksum != self.ebins_checksum:
      # self.ebins_checksum = ebins_checksum
      # for Z in self.elements:
        # #if Z=='temperatures': continue

        # for z1 in self.spectra[Z].keys():


          # self.spectra[Z][z1].set_ebins(self.ebins, ebins_checksum=self.ebins_checksum)

  # def calc_spectrum(self, kappa, T, ebins, abundset = 'angr', abund = 1.0, \
                    # elements=[1,2,6,7,8,10,12,13,14,16,18,20,26,28]):
    # """
    # Calculate a spectrum

    # PARAMETERS
    # ----------
    # kappa : float
      # kappa coefficient (>=2)
    # T : float
      # electron temperature (K)
    # ebins : array(float)
      # energy bins edges for spectrum (keV)
    # abundset : str
      # Abundance set (4 letter string from XSPEC)
    # abund : float of dict of floats
      # Abundance to apply for each element

    # RETURNS
    # -------
    # spectum : array(float)
      # resulting spectrum in photons bin-1 s-1 cm^3
    # """

    # # first, calculate the ionization balance
    # ionbal = self.calc_ionbal(kappa, T, elements)

    # # set abundances
    # if type(abund)==float:
      # for Z in self.elements:
        # self.abund[Z] = abund *modelabund(Z, abundset)
    # elif type(abund)==dict:
      # for Z in abund.keys():
        # self.abund[Z] = abund[Z]*modelabund(Z, abundset)

    # # get the kapp coefficients
    # tkappa, ckappa = self.get_kappa_coeffts( kappa, T)

    # # filter out of range ones
    # ckappa = ckappa[(tkappa >= 1e4) & (tkappa <= 1e9)]
    # tkappa = tkappa[(tkappa >= 1e4) & (tkappa <= 1e9)]

    # # for each temperature, calculate a spectrum
    # spec = numpy.zeros(len(ebins)-1)

    # for ik, tk in enumerate(tkappa):
      # ck=ckappa[ik]
      # for Z in ionbal.keys():
        # z1list = numpy.where(ionbal[Z] > MIN_IONBAL)[0]+1
        # if not Z in self.spectra.keys():
            # self.spectra[Z] = {}
        # for z1 in z1list:
          # if not z1 in self.spectra[Z].keys():

            # self.spectra[Z][z1] = IonSpectrum(Z, z1, self.spectra['temperatures'], self.linedata, self.compdata)

          # s = self.spectra[Z][z1].return_spectrum(tk, ebins, \
                                          # ebins_checksum = hashlib.md5(ebins).hexdigest(),\
                                          # scalefactor = self.abund[Z]*ionbal[Z][z1-1]*ck)

          # spec+= s * ck * self.abund[Z] * ionbal[Z][z1-1]

    # return spec




# class IonSpectrum():
  # """
  # Holds the details of the ion at all temperatures
  # """

  # def __init__(self, Z, z1, Tlist, linedata, cocodata):
    # self.Tlist = Tlist
    # self.Z = Z
    # self.z1 = z1
    # self.ionspectrumlist = {}
    # for iT in range(len(Tlist)):
      # self.ionspectrumlist[iT] = IonSpectrum_onetemp(Z, z1, self.Tlist[iT], linedata[iT+2].data, cocodata[iT+2].data)




  # def return_spectrum(self, T, ebins, ebins_checksum=False, scalefactor=1.0,\
                      # temperature=False, broadenlimit=1e-18):

    # """
    # Return the specturm at this temperatures
    # """

    # # Find the temperature I care about
    # ilo = max(numpy.where(self.Tlist < T)[0])
    # ihi = ilo + 1


    # r1 = 1- (T-self.Tlist[ilo])/(self.Tlist[ihi]-self.Tlist[ilo])
    # r2 = 1- r1



    # # great!

    # spec = self.ionspectrumlist[ilo].return_spectrum(ebins,\
                                                  # ebins_checksum=ebins_checksum, \
                                                  # scalefactor=1.0,\
                                                  # temperature=temperature, \
                                                  # broadenlimit=broadenlimit)*r1

    # spec += self.ionspectrumlist[ihi].return_spectrum(ebins,\
                                                  # ebins_checksum=ebins_checksum, \
                                                  # scalefactor=1.0,\
                                                  # temperature=temperature, \
                                                  # broadenlimit=broadenlimit)*r2

    # return spec




# class IonSpectrum_onetemp():
  # """
  # Holds the spectrum details for the ion at one temperature

  # """

  # def __init__(self, Z, z1, T, linedata, cocodata):

    # # store things
    # self.Z = Z
    # self.z1 = z1
    # self.T = T
    # lines = linedata[(linedata['Element'] == Z) &
                    # (linedata['Ion_Drv'] == z1)]


    # self.linelist = numpy.zeros(len(lines), dtype=numpy.dtype({'names':['energy','epsilon'],\
                                                     # 'formats':[float, float]}))
    # self.linelist['energy'] = const.HC_IN_KEV_A/lines['lambda']
    # self.linelist['epsilon'] = lines['epsilon']

    # # get the continuum information



    # c = numpy.where((cocodata['Z']==Z) & (cocodata['rmJ'] == z1))[0]
    # # set up to store continuum & coco files




    # n =cocodata['N_Cont'][c[0]]
    # self.cont=Continuum(cocodata['E_Cont'][c[0]][:n],\
                        # cocodata['Continuum'][c[0]][:n], \
                        # 'continuum')

    # n =cocodata['N_Pseudo'][c[0]]
    # self.pseudo=Continuum(cocodata['E_Pseudo'][c[0]][:n],\
                        # cocodata['Pseudo'][c[0]][:n], \
                        # 'pseudo')



    # # store the checksum so we know if we have to recalc later
    # self.ebins_checksum = False






  # def return_spectrum(self, ebins, ebins_checksum=False, scalefactor=1.0,\
                      # temperature=False, broadenlimit=1e-18):
    # """
    # Return the spectrum of the ion on the grid ebins

    # PARAMETERS
    # ----------
    # ebins : array(float)
      # energy bins in keV
    # ebins_checksum : str
      # checksum for ebins, to compare with earlier versions
    # scalefactor : float
      # multiply all emissivities by this before deciding if they are
      # above or below the broadening threshold
    # temperature : float
      # broaden all lines with emissivity*scalefactor > broadenlimit with
      # Gaussians reflecting this electron temperature (Kelvin)
    # broadenlimit : float
      # lines with emissivity > this will be broadened.

    # RETURNS
    # -------
    # spectrum : float(array)
      # emissivity in photons s-1 bin-1. Note scalefactor is  *NOT* applied to this
    # """

    # if ((ebins_checksum == False ) | \
        # (ebins_checksum != self.ebins_checksum)):
      # # need to recalculate the continuum spectrum

      # self.continuum = self.cont.return_spectrum(ebins) +\
                       # self.pseudo.return_spectrum(ebins)


    # # now do the lines!
    # spec = numpy.zeros(len(ebins)-1)

    # ### I HAVE NOT YET IMPLEMENTED BROADENING


# #    if temperature == True:
# #      # we are doing some broadening
# #      emiss = self.linelist['epsilon'] * scalefactor
# #      ibroad = emiss > broadenlimit
# #      s,z = numpy.histogram(self.linelist['energy'][~ibroad], \
# #                            bins=ebins, \
# #                            weights = self.linelist['epsilon'][~ibroad])
# #      spec += s
# #
# #      broadencoefft = temperature *const.ERG_KEV/ pyatomdb.atomic.Z_to_mass(self.Z)* const.AMUKG
# #      for l in self.linelist[ibroad]:
# #        # add each line with broadening


 # #   else:
    # s,z = numpy.histogram(self.linelist['energy'], \
                          # bins=ebins, \
                          # weights = self.linelist['epsilon'])
    # spec += s


    # spec += self.continuum

    # return spec



# class Continuum():
  # """
  # Class for holding a continuum for one ion

  # """

  # def __init__(self, E, C, conttype):
    # """
    # E : array(float)
    # C : array(float)
    # conttype : string
      # "psuedo" or "continuum"
    # """

    # self.E = E
    # self.C = C
    # self.conttype = conttype



  # def return_spectrum(self, ebins):
    # """
    # Return the spectrum of the ion on the grid ebins

    # PARAMETERS
    # ----------
    # ebins : array(float)
      # The bin edges, in keV

    # RETURNS
    # -------
    # spectrum : array(float)
      # The emissivity spectrum in ph cm^3 s^-1 bin^-1
    # """

    # #if ((not(ebins_checksum)) | (ebins_checksum != self.ebins_checksum)):
    # spec = self.expand_spectrum(ebins)
    # self.ebins_checksum = hashlib.md5(ebins).hexdigest()
    # return spec




  # def expand_spectrum(self, eedges):

    # """
    # Code to expand the compressed continuum onto a series of bins.

    # Parameters
    # ----------
    # eedges : float(array)
      # The bin edges for the spectrum to be calculated on, in units of keV

    # Returns
    # -------
    # float(array)
      # len(eedges)-1 array of continuum emission, in units of \
      # photons cm^3 s^-1 bin^-1
    # """
    # import scipy.integrate
    # n=len(self.E)
    # if n==2:
      # return 0.0
  # # ok. So. Sort this.
    # E_all = numpy.append(self.E, eedges)
    # cont_tmp = numpy.interp(eedges, self.E, self.C)
    # C_all = numpy.append(self.C, cont_tmp)

    # iord = numpy.argsort(E_all)

  # # order the arrays
    # E_all = E_all[iord]
    # C_all = C_all[iord]

    # ihi = numpy.where(iord>=n)[0]
    # cum_cont = scipy.integrate.cumtrapz(C_all, E_all, initial=0)
    # C_out = numpy.zeros(len(eedges))
    # C_out = cum_cont[ihi]

    # cont = C_out[1:]-C_out[:-1]
    # return cont


# def modelabund(Z, abundset):
  # modelabund={}
  # modelabund['angr'] = numpy.zeros(31)
  # modelabund['angr'][1]=1.00e+00
  # modelabund['angr'][2]=9.77e-02
  # modelabund['angr'][3]=1.45e-11
  # modelabund['angr'][4]=1.41e-11
  # modelabund['angr'][5]=3.98e-10
  # modelabund['angr'][6]=3.63e-04
  # modelabund['angr'][7]=1.12e-04
  # modelabund['angr'][8]=8.51e-04
  # modelabund['angr'][9]=3.63e-08
  # modelabund['angr'][10]=1.23e-04
  # modelabund['angr'][11]=2.14e-06
  # modelabund['angr'][12]=3.80e-05
  # modelabund['angr'][13]=2.95e-06
  # modelabund['angr'][14]=3.55e-05
  # modelabund['angr'][15]=2.82e-07
  # modelabund['angr'][16]=1.62e-05
  # modelabund['angr'][17]=3.16e-07
  # modelabund['angr'][18]=3.63e-06
  # modelabund['angr'][19]=1.32e-07
  # modelabund['angr'][20]=2.29e-06
  # modelabund['angr'][21]=1.26e-09
  # modelabund['angr'][22]=9.77e-08
  # modelabund['angr'][23]=1.00e-08
  # modelabund['angr'][24]=4.68e-07
  # modelabund['angr'][25]=2.45e-07
  # modelabund['angr'][26]=4.68e-05
  # modelabund['angr'][27]=8.32e-08
  # modelabund['angr'][28]=1.78e-06
  # modelabund['angr'][29]=1.62e-08
  # modelabund['angr'][30]=3.98e-08

  # modelabund['aspl'] = numpy.zeros(31)
  # modelabund['aspl'][1]= 1.00e+00
  # modelabund['aspl'][2]= 8.51e-02
  # modelabund['aspl'][3]= 1.12e-11
  # modelabund['aspl'][4]= 2.40e-11
  # modelabund['aspl'][5]= 5.01e-10
  # modelabund['aspl'][6]= 2.69e-04
  # modelabund['aspl'][7]= 6.76e-05
  # modelabund['aspl'][8]= 4.90e-04
  # modelabund['aspl'][9]= 3.63e-08
  # modelabund['aspl'][10]=8.51e-05
  # modelabund['aspl'][11]=1.74e-06
  # modelabund['aspl'][12]=3.98e-05
  # modelabund['aspl'][13]=2.82e-06
  # modelabund['aspl'][14]=3.24e-05
  # modelabund['aspl'][15]=2.57e-07
  # modelabund['aspl'][16]=1.32e-05
  # modelabund['aspl'][17]=3.16e-07
  # modelabund['aspl'][18]=2.51e-06
  # modelabund['aspl'][19]=1.07e-07
  # modelabund['aspl'][20]=2.19e-06
  # modelabund['aspl'][21]=1.41e-09
  # modelabund['aspl'][22]=8.91e-08
  # modelabund['aspl'][23]=8.51e-09
  # modelabund['aspl'][24]=4.37e-07
  # modelabund['aspl'][25]=2.69e-07
  # modelabund['aspl'][26]=3.16e-05
  # modelabund['aspl'][27]=9.77e-08
  # modelabund['aspl'][28]=1.66e-06
  # modelabund['aspl'][29]=1.55e-08
  # modelabund['aspl'][30]=3.63e-08

  # modelabund['feld'] = numpy.zeros(31)
  # modelabund['feld'][1]= 1.00e+00
  # modelabund['feld'][2]= 9.77e-02
  # modelabund['feld'][3]= 1.26e-11
  # modelabund['feld'][4]= 2.51e-11
  # modelabund['feld'][5]= 3.55e-10
  # modelabund['feld'][6]= 3.98e-04
  # modelabund['feld'][7]= 1.00e-04
  # modelabund['feld'][8]= 8.51e-04
  # modelabund['feld'][9]= 3.63e-08
  # modelabund['feld'][10]=1.29e-04
  # modelabund['feld'][11]=2.14e-06
  # modelabund['feld'][12]=3.80e-05
  # modelabund['feld'][13]=2.95e-06
  # modelabund['feld'][14]=3.55e-05
  # modelabund['feld'][15]=2.82e-07
  # modelabund['feld'][16]=1.62e-05
  # modelabund['feld'][17]=3.16e-07
  # modelabund['feld'][18]=4.47e-06
  # modelabund['feld'][19]=1.32e-07
  # modelabund['feld'][20]=2.29e-06
  # modelabund['feld'][21]=1.48e-09
  # modelabund['feld'][22]=1.05e-07
  # modelabund['feld'][23]=1.00e-08
  # modelabund['feld'][24]=4.68e-07
  # modelabund['feld'][25]=2.45e-07
  # modelabund['feld'][26]=3.24e-05
  # modelabund['feld'][27]=8.32e-08
  # modelabund['feld'][28]=1.78e-06
  # modelabund['feld'][29]=1.62e-08
  # modelabund['feld'][30]=3.98e-08

  # modelabund['aneb'] = numpy.zeros(31)
  # modelabund['aneb'][1]= 1.00e+00
  # modelabund['aneb'][2]= 8.01e-02
  # modelabund['aneb'][3]= 2.19e-09
  # modelabund['aneb'][4]= 2.87e-11
  # modelabund['aneb'][5]= 8.82e-10
  # modelabund['aneb'][6]= 4.45e-04
  # modelabund['aneb'][7]= 9.12e-05
  # modelabund['aneb'][8]= 7.39e-04
  # modelabund['aneb'][9]= 3.10e-08
  # modelabund['aneb'][10]=1.38e-04
  # modelabund['aneb'][11]=2.10e-06
  # modelabund['aneb'][12]=3.95e-05
  # modelabund['aneb'][13]=3.12e-06
  # modelabund['aneb'][14]=3.68e-05
  # modelabund['aneb'][15]=3.82e-07
  # modelabund['aneb'][16]=1.89e-05
  # modelabund['aneb'][17]=1.93e-07
  # modelabund['aneb'][18]=3.82e-06
  # modelabund['aneb'][19]=1.39e-07
  # modelabund['aneb'][20]=2.25e-06
  # modelabund['aneb'][21]=1.24e-09
  # modelabund['aneb'][22]=8.82e-08
  # modelabund['aneb'][23]=1.08e-08
  # modelabund['aneb'][24]=4.93e-07
  # modelabund['aneb'][25]=3.50e-07
  # modelabund['aneb'][26]=3.31e-05
  # modelabund['aneb'][27]=8.27e-08
  # modelabund['aneb'][28]=1.81e-06
  # modelabund['aneb'][29]=1.89e-08
  # modelabund['aneb'][30]=4.63e-08

  # modelabund['grsa'] = numpy.zeros(31)
  # modelabund['grsa'][1]= 1.00e+00
  # modelabund['grsa'][2]= 8.51e-02
  # modelabund['grsa'][3]= 1.26e-11
  # modelabund['grsa'][4]= 2.51e-11
  # modelabund['grsa'][5]= 3.55e-10
  # modelabund['grsa'][6]= 3.31e-04
  # modelabund['grsa'][7]= 8.32e-05
  # modelabund['grsa'][8]= 6.76e-04
  # modelabund['grsa'][9]= 3.63e-08
  # modelabund['grsa'][10]=1.20e-04
  # modelabund['grsa'][11]=2.14e-06
  # modelabund['grsa'][12]=3.80e-05
  # modelabund['grsa'][13]=2.95e-06
  # modelabund['grsa'][14]=3.55e-05
  # modelabund['grsa'][15]=2.82e-07
  # modelabund['grsa'][16]=2.14e-05
  # modelabund['grsa'][17]=3.16e-07
  # modelabund['grsa'][18]=2.51e-06
  # modelabund['grsa'][19]=1.32e-07
  # modelabund['grsa'][20]=2.29e-06
  # modelabund['grsa'][21]=1.48e-09
  # modelabund['grsa'][22]=1.05e-07
  # modelabund['grsa'][23]=1.00e-08
  # modelabund['grsa'][24]=4.68e-07
  # modelabund['grsa'][25]=2.45e-07
  # modelabund['grsa'][26]=3.16e-05
  # modelabund['grsa'][27]=8.32e-08
  # modelabund['grsa'][28]=1.78e-06
  # modelabund['grsa'][29]=1.62e-08
  # modelabund['grsa'][30]=3.98e-08

  # modelabund['wilm'] = numpy.zeros(31)
  # modelabund['wilm'][1]= 1.00e+00
  # modelabund['wilm'][2]= 9.77e-02
  # modelabund['wilm'][3]= 0.00
  # modelabund['wilm'][4]= 0.00
  # modelabund['wilm'][5]= 0.00
  # modelabund['wilm'][6]= 2.40e-04
  # modelabund['wilm'][7]= 7.59e-05
  # modelabund['wilm'][8]= 4.90e-04
  # modelabund['wilm'][9]= 0.00
  # modelabund['wilm'][10]=8.71e-05
  # modelabund['wilm'][11]=1.45e-06
  # modelabund['wilm'][12]=2.51e-05
  # modelabund['wilm'][13]=2.14e-06
  # modelabund['wilm'][14]=1.86e-05
  # modelabund['wilm'][15]=2.63e-07
  # modelabund['wilm'][16]=1.23e-05
  # modelabund['wilm'][17]=1.32e-07
  # modelabund['wilm'][18]=2.57e-06
  # modelabund['wilm'][19]=0.00
  # modelabund['wilm'][20]=1.58e-06
  # modelabund['wilm'][21]=0.00
  # modelabund['wilm'][22]=6.46e-08
  # modelabund['wilm'][23]=0.00
  # modelabund['wilm'][24]=3.24e-07
  # modelabund['wilm'][25]=2.19e-07
  # modelabund['wilm'][26]=2.69e-05
  # modelabund['wilm'][27]=8.32e-08
  # modelabund['wilm'][28]=1.12e-06
  # modelabund['wilm'][29]=0.00
  # modelabund['wilm'][30]=0.00

  # modelabund['lodd'] = numpy.zeros(31)
  # modelabund['lodd'][1]= 1.00e+00
  # modelabund['lodd'][2]= 7.92e-02
  # modelabund['lodd'][3]= 1.90e-09
  # modelabund['lodd'][4]= 2.57e-11
  # modelabund['lodd'][5]= 6.03e-10
  # modelabund['lodd'][6]= 2.45e-04
  # modelabund['lodd'][7]= 6.76e-05
  # modelabund['lodd'][8]= 4.90e-04
  # modelabund['lodd'][9]= 2.88e-08
  # modelabund['lodd'][10]=7.41e-05
  # modelabund['lodd'][11]=1.99e-06
  # modelabund['lodd'][12]=3.55e-05
  # modelabund['lodd'][13]=2.88e-06
  # modelabund['lodd'][14]=3.47e-05
  # modelabund['lodd'][15]=2.88e-07
  # modelabund['lodd'][16]=1.55e-05
  # modelabund['lodd'][17]=1.82e-07
  # modelabund['lodd'][18]=3.55e-06
  # modelabund['lodd'][19]=1.29e-07
  # modelabund['lodd'][20]=2.19e-06
  # modelabund['lodd'][21]=1.17e-09
  # modelabund['lodd'][22]=8.32e-08
  # modelabund['lodd'][23]=1.00e-08
  # modelabund['lodd'][24]=4.47e-07
  # modelabund['lodd'][25]=3.16e-07
  # modelabund['lodd'][26]=2.95e-05
  # modelabund['lodd'][27]=8.13e-08
  # modelabund['lodd'][28]=1.66e-06
  # modelabund['lodd'][29]=1.82e-08
  # modelabund['lodd'][30]=4.27e-08

  # return modelabund[abundset][Z]/modelabund['angr'][Z]
