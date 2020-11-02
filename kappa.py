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
#__version__='1.1.1'

# November 2nd 2020 ARF: Several updates:
#  - Fixed error in hahnsavin.fits file affecting 5.2<=kappa<7.3
#    (wrong value was in published paper, right value was in accompanying IDL code)
#  - Recoded much of the Session and Spectrum process to use
#    the same routines as the NEISpectrum and NEISession classes in pyatomdb
#  - Made the return_line_emissivity and return_linelist functions work
#  - Made doline, docont and dopseudo keywords work
__version__='1.2.0'


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


    self.SessionType='Kappa'
    self._session_initialise1(linefile, cocofile, elements, abundset)

    # define the directories & other data files
    if kappadir==None:
      # set to the directory where this file sits
      self.kappadir = os.path.dirname(os.path.realpath(__file__))
      os.environ['ATOMDBKAPPA'] = self.kappadir
    else:
      os.environ['ATOMDBKAPPA'] = kappadir

    self.hsdatafile = os.path.expandvars(hsdatafile)
    self.ionrecdatafile = os.path.expandvars(ionrecdatafile)

    self.spectra=_KappaSpectrum(self.linedata, self.cocodata, \
                                self.hsdatafile, self.ionrecdatafile,
                                elements = self.elements)

    self._session_initialise2()




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


  def return_linelist(self, Te, kappa, specrange, specunit='A', \
                               teunit='keV', apply_aeff=False, \
                               develop=False):
    """
    Get the list of line emissivities vs wavelengths


    Parameters
    ----------
    Te : float
      Temperature in keV or K
    kappa : float
      Non Maxwellian kappa parameter. Must be > 1.5.
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



    kT = util.convert_temp(Te, teunit, 'keV')

    el_list = self.elements
    ab = {}
    for Z in el_list:
      ab[Z] = self.abund[Z]*self.abundsetvector[Z]



    s= self.spectra.return_linelist(kT, kappa, \
                                    specrange=specrange, teunit='keV',\
                                    specunit=specunit, elements=self.elements,\
                                    abundance = ab, log_interp=True)

    # do the response thing
    #resp  = s.response()

    if apply_aeff == True:

      epsilon_aeff =  self._apply_linelist_aeff(s, specunit, apply_binwidth)

      s['Epsilon_Err'] = epsilon_aeff
    return(s)


  def return_line_emissivity(self, Telist, kappalist, Z, z1, up, lo, specunit='A',
                             teunit='keV',
                             apply_aeff=False, apply_abund=True,\
                             log_interp = True):
    """
    Return the emissivity of a line at kT, tau. Assumes ionization from neutral for now


    Parameters
    ----------
    Telist : float or array(float)
      Temperature(s) in keV or K
    kappa : float or array(float)
      Non Maxwellian kappa parameter. Must be > 1.5.
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
      Units of Te (kev or K, default keV)
    abundance : float
      Abundance to multiply the emissivity by
    log_interp : bool
      Perform linear interpolation on a logT/logEpsilon grid (default), or linear.

    Returns
    -------
    ret : dict
      Dictionary containing:
      Te, kappa, teunit: as input
      wavelength : line wavelength (A)
      energy : line energy (keV)
      epsilon : emissivity in ph cm^3 s-1 (or ph cm^5 s^-1 if apply_aeff=True)
                first index is temperature, second is kappa. If Te or kappa was
                supplied as a scalar, then that index is removed

    """

    Tevec, Teisvec = util.make_vec(Telist)
    kappavec, kappaisvec = util.make_vec(kappalist)


    kTlist = util.convert_temp(Tevec, teunit, 'keV')

    eps = numpy.zeros([len(Tevec), len(kappavec)])
    ret={}
    ret['wavelength'] = None


    if apply_abund:
      ab = self.abund[Z]*self.abundsetvector[Z]
    else:
      ab = 1.0

    for ikappa, kappa in enumerate(kappavec):
      for ikT, kT in enumerate(kTlist):
        e, lam = self.spectra.return_line_emissivity(kT, kappa, Z, z1, \
                                                     up, lo, \
                                                     specunit='A', \
                                                     teunit='keV', \
                                                     abundance=ab)

        eps[ikT, ikappa] = e
        if lam != False:
          ret['wavelength'] = lam * 1.0
        else:
          ret['wavelength'] = None

    ret['Te'] = Telist
    ret['kappa'] = kappalist
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

    if not kappaisvec:
      eps=eps[:,0]
      if not Teisvec:
        eps = eps[0]
    else:
      if not Teisvec:
        eps = eps[0,:]

    ret['epsilon'] = eps

    return ret


  def return_spectrum(self,  Te, kappa, teunit='keV',\
                      log_interp=True):
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
    kappa : float
      Non Maxwellian kappa parameter. Must be > 1.5.
    teunit : {'keV' , 'K'}
      Units of te (kev or K, default keV)
    log_interp : bool
      Interpolate between temperature on a log-log scale (default).
      Otherwise linear



    Returns
    -------
    spectrum : array(float)
      The spectrum in photons cm^5 s^-1 bin^-1, with the response, or
      photons cm^3 s^-1 bin^-1 if raw is set.
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

    self.spectra.dolines = self.dolines
    self.spectra.dopseudo = self.dopseudo
    self.spectra.docont = self.docont


    s= self.spectra.return_spectrum(Te, kappa, teunit=teunit, elements = el_list, \
                                    abundances=ab, log_interp=True,\
                                    broaden_object=self.cdf)

    ss = self._apply_response(s)

    return ss

























class _KappaSpectrum(spectrum._NEISpectrum):
  """
  A class holding the emissivity data for NEI emission, and returning
  spectra

  Parameters
  ----------
  linefile : string or HDUList, optional
    The line emissivity data file (either name or already open)
  cocofile : string or HDUList, optional
    The continuum emissivity data file (either name or already open)
  hsdatafile : string
    Name of FITS file with the H&S coefficient data.
  ionrecdatafile : string
    Name of FITS file with the abbreviated ionization and recombination coefficient data.
  elements : arraylike(int), optional
    The atomic number of elements to include (default all)

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

    havepicklefile = False
    if os.path.isfile(picklefname):
      havepicklefile = True

    if havepicklefile:
      try:
        self.spectra = pickle.load(open(picklefname,'rb'))
        self.kTlist = self.spectra['kTlist']
      except AttributeError:
        havepicklefile=False
        print("pre-stored data in %s is out of date. This can be caused by updates to the data "%(picklefname)+
              "or, more likely, changes to pyatomdb. Regenerating...")
    else:
        # delete the old file
        if os.path.isfile(picklefname):
          os.remove(picklefname)

    if not havepicklefile:
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
            self.spectra[ihdu][Z][z1]=_ElementSpectrum(ldat[isgood],\
                                                  ccdat[0], Z, z1_drv=z1)


      pickle.dump(self.spectra, open(picklefname,'wb'))


    self.logkTlist=numpy.log(self.kTlist)

    # now repeat for hahn savin data
    self.hsdata = hs_data(hsdatafile)

    # now repeat for ionization and recombination
    self.ionrecdata = ir_data(irfile = ionrecdatafile, elements=self.elements)



  def _calc_ionrec_rate(self, tkappa, ckappa, elements):
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



  # def return_oneT_spectrum(self, Te, Z, z1, epslimit, teunit='keV', log_interp=True,\
                           # broaden_object=False, ikT=False, f=False):
    # """
    # return a single element, single ion, spectrum, interpolating
    # appropriately between neighboring temperature bins

    # """
    # T = util.convert_temp(Te, teunit, 'K')
    # kT = util.convert_temp(Te, teunit, 'keV')


    # # Recalc fractions if required
    # if (type(ikT)==bool) | (type(f)==bool):
      # ikT, f = self.get_nearest_Tindex(kT, teunit='keV',  log_interp=log_interp)


    # # ok, get the spectra
    # stot=0.0
    # for i in range(len(ikT)):

      # # get the spectrum
      # sss = self.spectra[ikT[i]][Z][z1].return_spectrum(self.ebins,\
                                   # kT,\
                                   # ebins_checksum = self.ebins_checksum,\
                                   # thermal_broadening = self.thermal_broadening,\
                                   # broaden_limit = epslimit,\
                                   # velocity_broadening = self.velocity_broadening,\
                                   # broaden_object=broaden_object)
      # # add it appropriately
      # if log_interp:
        # stot += numpy.log(sss+const.MINEPSOFFSET)*f[i]
      # else:
        # stot +=sss*f[i]
    # # now handle the sum

    # stot = numpy.exp(stot)-const.MINEPSOFFSET*len(f)
    # stot[stot<0] = 0.0
    # return stot




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
    kappa : float
       kappa coefficient (>1.5)
    teunit : string
      Units of kT (keV by default, K also allowed)
    FIXME
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


    #ionrate, recrate = self.calc_ionrec_rate(tkappa_all, ckappa_all, elements)
    self._calc_ionbal(tkappa_all, ckappa_all, elements)

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


    stot = 0.0


    for ik, tk in enumerate(tkappa):
      ikT, f = self.get_nearest_Tindex(tk, teunit='K',  log_interp=log_interp)

      s={}
      s[0] = 0.0
      s[1] = 0.0


      for Z in elements:
        abund = abundances[Z]
        if abund > 0:

        # solve the ionization balance
     #   self.ionbal[Z] = apec.solve_ionbal(ionrate[Z], recrate[Z])
          ionfrac = self.ionbal[Z]

          for z1 in range(1, Z+2):

            if ionfrac[z1-1]>1e-10:

              # calculate minimum emissivitiy to broaden, accounting for ion
              # and element abundance.
              epslimit =  self.broaden_limit/(abund*ionfrac[z1-1])

              for i, iikT in enumerate(ikT):

                s[i] += self.spectra[ikT[0]][Z][z1].return_spectrum(self.ebins,\
                                  kT,\
                                  ebins_checksum = self.ebins_checksum,\
                                  thermal_broadening = self.thermal_broadening,\
                                  broaden_limit = epslimit,\
                                  velocity_broadening = self.velocity_broadening,\
                                  broaden_object=broaden_object,\
                                  dolines=self.dolines,\
                                  dopseudo=self.dopseudo,\
                                  docont=self.docont,\
                                  ) *\
                                  ionfrac[z1-1] * abund

      # merge the spectra
      smerge = self._merge_spectra_temperatures(f, s[0], s[1], log_interp)
      stot += smerge*ckappa[ik]

    return stot

  def _calc_ionbal(self, tkappa_all, ckappa_all, elements):
    ionrate, recrate = self._calc_ionrec_rate(tkappa_all, ckappa_all, elements)
    self.ionbal={}
    for Z in elements:
      # solve the ionization balance
      self.ionbal[Z] = apec.solve_ionbal(ionrate[Z], recrate[Z])

  def return_line_emissivity(self, Te, kappa, Z, z1, up, lo, specunit='A',
                             teunit='keV', abundance=1.0,
                             log_interp = True):
    """
    Return the emissivity of a line at kT, tau. Assumes ionization from neutral for now

    Parameters
    ----------
    Te : float
      Temperature in keV or K
    kappa : float
      Non Maxwellian kappa parameter. Must be > 1.5.
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
    log_interp : bool
      Interpolate between temperature on a log-log scale (default).
      Otherwise linear

    Returns
    -------
    Emissivity : float
      Emissivity in photons cm^3 s^-1
    spec : float
      Wavelength or Energy of line, depending on specunit
    """

    import collections

    kT = util.convert_temp(Te, teunit, 'keV')
    T = util.convert_temp(Te, teunit, 'K')

    # find the correct coefficients here
    tkappa_all, ckappa_all = self.hsdata.get_coeffts(kappa, T)
    self._calc_ionbal(tkappa_all, ckappa_all, [Z])
    # filter out of range ones
    ckappa = ckappa_all[(tkappa_all >= 1e4) & (tkappa_all <= 1e9)]
    if len(ckappa) < len(tkappa_all):
      print("Note: only using %i of %i requested Maxwellian components as they are inside the 10^4 to 10^9K range"%(len(ckappa), len(tkappa_all)))

    tkappa = tkappa_all[(tkappa_all >= 1e4) & (tkappa_all <= 1e9)]


    eps = 0.0
    lam = 0.0

    ionfrac = self.ionbal[Z]

    for ik, tk in enumerate(tkappa):
      ikT, f = self.get_nearest_Tindex(tk, teunit='K',  log_interp=log_interp)
      # find lines which match
      eps_tmp = 0.0
      for z1_drv in range(1,Z+2):
      # ions which don't exist get skipped
        if ionfrac[z1_drv-1] <= 1e-10: continue
        eps_in = numpy.zeros(len(ikT))

        for i, iikT in enumerate(ikT):
          llist = self.spectra[iikT][Z][z1_drv].return_linematch(Z,z1,up,lo)
          for line in llist:
            # add emissivity
            eps_in[i] += line['Epsilon']*ionfrac[z1_drv-1]
            lam = line['Lambda']
        # now merge
        eps_tmp += eps_in
      eps_x = 0.0
      if log_interp:
        for i in range(len(ikT)):
          eps_x += f[i]*numpy.log(eps_tmp[i]+const.MINEPSOFFSET)
        eps_tmp = (numpy.exp(eps_x)-const.MINEPSOFFSET)*abundance *ckappa[ik]
      else:
        for i in range(len(ikT)):
          eps_x += f[i]*eps_tmp[i]
        eps_tmp = eps_x*abundance *ckappa[ik]

      eps += eps_tmp

    if specunit == 'keV':
      lam = const.HC_IN_KEV_A/lam
    return eps, lam

  def return_linelist(self,  Te, kappa,\
                      teunit='keV', nearest=False, specrange=False,\
                      specunit='A', elements=False, abundance=False,\
                      log_interp=True):

    """
    Return the linelist of the element

    Parameters
    ----------

    Te : float
      Electron temperature (default, keV)
    kappa : float
      Non Maxwellian kappa parameter. Must be > 1.5.
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
    elements : iterable of int
      Elements to include, listed by atomic number. if not set, include all.
    abundance : dict(float)
      The abundances of each element, e.g. abund[6]=1.1 means multiply carbon
      abundance by 1.1.
    log_interp : bool
      Interpolate between temperature on a log-log scale (default).
      Otherwise linear

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
      elements=range(1,const.MAXZ_NEI+1)

    if abundance == False:
      abundance = {}
      for Z in elements:
        abundance[Z] = 1.0

    self._calc_ionbal(tkappa, ckappa, elements)

    linelist = numpy.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))

    # set up arrays to store by element lines
    s={}
    for Z in elements:
      s[Z] = numpy.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))

    for ik, tk in enumerate(tkappa):
      ikT, f = self.get_nearest_Tindex(tk, teunit='K',  log_interp=log_interp)



      for Z in elements:
        abund = abundance[Z]
        stmp={}
        stmp['Z'] = {}
        stmp['Z'][0] = numpy.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))
        stmp['Z'][1] = numpy.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))

        if abund > 0:

        # solve the ionization balance
     #   self.ionbal[Z] = apec.solve_ionbal(ionrate[Z], recrate[Z])
          ionfrac = self.ionbal[Z]

          for z1_drv in range(1, Z+2):

            if ionfrac[z1_drv-1]>1e-10:

              # calculate minimum emissivitiy to broaden, accounting for ion
              # and element abundance.
              epslimit =  self.broaden_limit/(abund*ionfrac[z1_drv-1])


              llist={}
              for i, iikT in enumerate(ikT):
                llist[i] = self.spectra[iikT][Z][z1_drv].return_linelist(specrange, specunit=specunit)

                llist[i]['Epsilon']*= ionfrac[z1_drv-1] * abund

                tmpl = numpy.zeros(len(llist[i]), dtype=apec.generate_datatypes('linelist_cie_spectrum'))
                for key in tmpl.dtype.names:
                  tmpl[key] = llist[i][key]
                stmp['Z'][i] = numpy.append(stmp['Z'][i], tmpl)

        #merge all the lines of the element at each temperature
        stmp['Z']['sum']=self._merge_linelists_temperatures(f, stmp['Z'][0], stmp['Z'][1], \
                                                     log_interp, \
                                                     by_ion_drv=False)
        # normalize for kappa C fraction
        stmp['Z']['sum']['Epsilon'] *= ckappa[ik]

        # append to the overall linelist for the element
        s[Z]=numpy.append(s[Z], stmp['Z']['sum'])

    # create a return linelist - all elements, removing duplicates
    s_out = numpy.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))
    for Z in elements:
      s_out=numpy.append(s_out, self._merge_linelist_duplicates(s[Z], by_ion_drv=False))

    return s_out

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


