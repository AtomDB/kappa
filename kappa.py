import numpy, pickle, pyatomdb, os, hashlib
import astropy.io.fits as pyfits

"""
This modules is designed to generate kappa specta.
Method is:
  - look up the coefficients from the Hahn & Savin paper,
  - Use these to assemble ionization and recombination rates
  - Solve ion balance
  - Make spectra for the relevant ions at temperatures defined by Hahn & Savin
  - Sum spectra to get... well, a spectrum.

"""


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
class hs_data():
  """
  Class to store the read in Hahn & Savin Data.
  Can then be queried to spit out the relevant rates, etc.

  """

  def __init__(self):
    """
    Read in the data
    """

    # for now, this is hardwired to load a pickle file. This is not
    # ideal, will be switched over to proper FITS files when happy
    # with format, etc.
    tmp = pyatomdb.pyfits.open('hahnsavin.fits')

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
# GREAT!


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


class irdata():
#-------------------------------------------------------------------------------

  def __init__(self, elements=[1,2,6,7,8,10,12,13,14,16,18,20,26,28],\
               irfile = 'kappa_ir.fits'):

    # load up the kappa coefficients
    self.elements=elements
    ionrec_data = {}
    ir = pyatomdb.pyfits.open(irfile)
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
      chi  = dE / (pyatomdb.const.KBOLTZ*Tvec)


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
      calc_type = pyatomdb.const.EI_UPSILON


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

    if ((cidat['par_type']>pyatomdb.const.INTERP_I_UPSILON) & \
        (cidat['par_type']<=pyatomdb.const.INTERP_I_UPSILON+20)):


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


    elif ((cidat['par_type']>pyatomdb.const.INTERP_I_UPSILON) & \
        (cidat['par_type']<=pyatomdb.const.INTERP_I_UPSILON+20)):
      ci = calc_maxwell_rate(cidat['PAR_TYPE'],\
                         cidat['MIN_TEMP'],\
                         cidat['MAX_TEMP'],\
                                 cidat['TEMPERATURE'],\
                                 cidat['IONREC_PAR'],\
                                 ionpot/1e3, T, Z, degl, degu)


    elif ((cidat['par_type']>pyatomdb.const.CI_DERE) &\
          (cidat['par_type']<=pyatomdb.const.CI_DERE+20)):
      ionpot = self.ionrecdata[Z][z1]['IP_DERE']
      Tvec, wasTvec = make_vec(T)
      ci = pyatomdb.atomdb.calc_ionrec_ci(cidat, Tvec, extrap=True, ionpot=ionpot)

    else:
      Tvec, wasTvec = make_vec(T)
      ci = pyatomdb.atomdb.calc_ionrec_ci(cidat,Tvec, extrap=True)


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

    dr = pyatomdb.atomdb.calc_ionrec_dr(drdat, T, extrap=True)
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

    rr = pyatomdb.atomdb.calc_ionrec_rr(rrdat, T, extrap=True)
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

    ea = pyatomdb.atomdb.calc_ionrec_ea(eadat, T, extrap=True)
    return ea


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



class kappamodel():
  """
  Class for all things kappa
  """

  def __init__(self, elements=[1,2,6,7,8,10,12,13,14,16,18,20,26,28],\
               linefile = "$ATOMDB/apec_nei_line.fits",\
               compfile = "$ATOMDB/apec_nei_comp.fits"):

    # load up the kappa coefficients
    self.hs_data = hs_data()
    self.elements = elements
    self.abund = {}
    for Z in self.elements:
      self.abund[Z] = 1.0
    self.ionrec_data = irdata(elements = self.elements)

    # for spectra
    self.ebins = False
    self.ebins_checksum = False
    self.spectra = {}

    self.linedata = pyatomdb.pyfits.open(os.path.expandvars(linefile))
    self.compdata = pyatomdb.pyfits.open(os.path.expandvars(compfile))


    self.spectra['temperatures'] = self.linedata[1].data['kT']/pyatomdb.const.KBOLTZ
    for Z in self.elements:
      self.spectra[Z] = {}
      #for z1 in range(1, Z+2):
      #  self.spectra[Z][z1] = IonSpectrum(Z, z1, self.spectra['temperatures'], self.linedata, self.compdata)
    print("Kappa model ready for use. Note that the first time a spectrum "+\
          "is calculated it can take up to a  minute to finish. It gets much "+\
          "faster after that, I promise.")


  def get_kappa_coeffts(self, kappa, T):
    """
    Return the kappa coefficients and temperatures at desired temperature

    PARAMETERS
    ----------
    kappa : float
      kappa coefficient (>1.5)
    T : float
      temperature (K)

    RETURNS
    -------
    Tkappa : array(float)
      temperatures required for kappa calculation (K)
    ckappa : array(float)
      coefficients at these temperatures
    """
    Tkappa, ckappa = self.hs_data.get_coeffts(kappa,T)

    return(Tkappa, ckappa)

  def calc_ionrec_rate(self, kappa, T):
    """
    Calculate the ionization and recombination rates for a kappa
    distribution, by summing maxwellians

    PARAMETERS
    ----------
    kappa : float
      kappa coefficient (>1.5)
    T : float
      temperature (K)

    RETURNS
    -------
    ionrate : dict
      e.g. ionrate[16] is the ionization rate coefft for sulphur 1 through 17, in cm^3 s-1
    recrate : dict
      e.g. recrate[16] is the recombiation rate coefft for sulphur 1 through 17, in cm^3 s-1
    """

    tkappa, ckappa = self.get_kappa_coeffts( kappa, T)
    # filter OOB kappa:
#    ckappa = ckappa[ (tkappa >=1e4) & (tkappa <= 1e9)]
#    tkappa = tkappa[ (tkappa >=1e4) & (tkappa <= 1e9)]

    ircoeffts = self.ionrec_data.get_ir_rate(tkappa)


    ionrate = {}
    recrate = {}


    for Z in self.elements:
      ionrate[Z] = numpy.zeros(Z)
      recrate[Z] = numpy.zeros(Z)
      for z1 in range(1,Z+1):
        ionrate[Z][z1-1]=sum( ircoeffts[Z]['ion'][z1-1,:]*ckappa)
        recrate[Z][z1-1]=sum( ircoeffts[Z]['rec'][z1-1,:]*ckappa)

    self.tkappa = tkappa
    self.ckappa = ckappa
    return ionrate, recrate


  def calc_ionbal(self, kappa, T):
    """
    Calculate the ionization balance for a kappa distribution by summing
    Maxwellians

    PARAMETERS
    ----------
    kappa : float
      kappa coefficient (>1.5)
    T : float
      temperature (K)

    RETURNS
    -------
    ionbal : dict
      e.g. ionbal[16] is a 17 element array with the fractional abundance of each ion of S, starting with neutral
    """

    ionrate, recrate = self.calc_ionrec_rate(kappa, T)


    ionbal = {}
    for Z in self.elements:
      ionbal[Z] = pyatomdb.apec.solve_ionbal(ionrate[Z], recrate[Z])
    self.ionbal = ionbal


    return ionbal


  def set_ebins(self, ebins, ebins_checksum = False):
    """
    Set the energy bins for the spectrum being returned.

    PARAMETERS
    ----------
    ebins : array(float)
      Energy bin edges (keV)
    ebins_checksum : string, optional
      The hex digest of the md5 sum of ebins. Used to check for changes.
    """

    if ebins_checksum == False:
      ebins_checksum = hashlib.md5(ebins).hexdigest()

    self.ebins = ebins

    if ebins_checksum != self.ebins_checksum:
      self.ebins_checksum = ebins_checksum
      for Z in self.elements:
        #if Z=='temperatures': continue

        for z1 in self.spectra[Z].keys():


          self.spectra[Z][z1].set_ebins(self.ebins, ebins_checksum=self.ebins_checksum)

  def calc_spectrum(self, kappa, T, ebins, abundset = 'angr', abund = 1.0):
    """
    Calculate a spectrum

    PARAMETERS
    ----------
    kappa : float
      kappa coefficient (>=2)
    T : float
      electron temperature (K)
    ebins : array(float)
      energy bins edges for spectrum (keV)
    abundset : str
      Abundance set (4 letter string from XSPEC)
    abund : float of dict of floats
      Abundance to apply for each element

    RETURNS
    -------
    spectum : array(float)
      resulting spectrum in photons bin-1 s-1 cm^3
    """

    # first, calculate the ionization balance
    ionbal = self.calc_ionbal(kappa, T)

    # set abundances
    if type(abund)==float:
      for Z in self.elements:
        self.abund[Z] = abund *modelabund(Z, abundset)
    elif type(abund)==dict:
      for Z in abund.keys():
        self.abund[Z] = abund[Z]*modelabund(Z, abundset)

    # get the kapp coefficients
    tkappa, ckappa = self.get_kappa_coeffts( kappa, T)

    # filter out of range ones
    ckappa = ckappa[(tkappa >= 1e4) & (tkappa <= 1e9)]
    tkappa = tkappa[(tkappa >= 1e4) & (tkappa <= 1e9)]

    # for each temperature, calculate a spectrum
    spec = numpy.zeros(len(ebins)-1)

    for ik, tk in enumerate(tkappa):
      ck=ckappa[ik]
      for Z in ionbal.keys():
        z1list = numpy.where(ionbal[Z] > MIN_IONBAL)[0]+1
        if not Z in self.spectra.keys():
            self.spectra[Z] = {}
        for z1 in z1list:
          if not z1 in self.spectra[Z].keys():

            self.spectra[Z][z1] = IonSpectrum(Z, z1, self.spectra['temperatures'], self.linedata, self.compdata)

          s = self.spectra[Z][z1].return_spectrum(tk, ebins, \
                                          ebins_checksum = hashlib.md5(ebins).hexdigest(),\
                                          scalefactor = self.abund[Z]*ionbal[Z][z1-1]*ck)

          spec+= s * ck * self.abund[Z] * ionbal[Z][z1-1]

    return spec




class IonSpectrum():
  """
  Holds the details of the ion at all temperatures
  """

  def __init__(self, Z, z1, Tlist, linedata, cocodata):
    self.Tlist = Tlist
    self.Z = Z
    self.z1 = z1
    self.ionspectrumlist = {}
    for iT in range(len(Tlist)):
      self.ionspectrumlist[iT] = IonSpectrum_onetemp(Z, z1, self.Tlist[iT], linedata[iT+2].data, cocodata[iT+2].data)




  def return_spectrum(self, T, ebins, ebins_checksum=False, scalefactor=1.0,\
                      temperature=False, broadenlimit=1e-18):

    """
    Return the specturm at this temperatures
    """

    # Find the temperature I care about
    ilo = max(numpy.where(self.Tlist < T)[0])
    ihi = ilo + 1


    r1 = 1- (T-self.Tlist[ilo])/(self.Tlist[ihi]-self.Tlist[ilo])
    r2 = 1- r1



    # great!

    spec = self.ionspectrumlist[ilo].return_spectrum(ebins,\
                                                  ebins_checksum=ebins_checksum, \
                                                  scalefactor=1.0,\
                                                  temperature=temperature, \
                                                  broadenlimit=broadenlimit)*r1

    spec += self.ionspectrumlist[ihi].return_spectrum(ebins,\
                                                  ebins_checksum=ebins_checksum, \
                                                  scalefactor=1.0,\
                                                  temperature=temperature, \
                                                  broadenlimit=broadenlimit)*r2

    return spec




class IonSpectrum_onetemp():
  """
  Holds the spectrum details for the ion at one temperature

  """

  def __init__(self, Z, z1, T, linedata, cocodata):

    # store things
    self.Z = Z
    self.z1 = z1
    self.T = T
    lines = linedata[(linedata['Element'] == Z) &
                    (linedata['Ion_Drv'] == z1)]


    self.linelist = numpy.zeros(len(lines), dtype=numpy.dtype({'names':['energy','epsilon'],\
                                                     'formats':[float, float]}))
    self.linelist['energy'] = pyatomdb.const.HC_IN_KEV_A/lines['lambda']
    self.linelist['epsilon'] = lines['epsilon']

    # get the continuum information



    c = numpy.where((cocodata['Z']==Z) & (cocodata['rmJ'] == z1))[0]
    # set up to store continuum & coco files




    n =cocodata['N_Cont'][c[0]]
    self.cont=Continuum(cocodata['E_Cont'][c[0]][:n],\
                        cocodata['Continuum'][c[0]][:n], \
                        'continuum')

    n =cocodata['N_Pseudo'][c[0]]
    self.pseudo=Continuum(cocodata['E_Pseudo'][c[0]][:n],\
                        cocodata['Pseudo'][c[0]][:n], \
                        'pseudo')



    # store the checksum so we know if we have to recalc later
    self.ebins_checksum = False






  def return_spectrum(self, ebins, ebins_checksum=False, scalefactor=1.0,\
                      temperature=False, broadenlimit=1e-18):
    """
    Return the spectrum of the ion on the grid ebins

    PARAMETERS
    ----------
    ebins : array(float)
      energy bins in keV
    ebins_checksum : str
      checksum for ebins, to compare with earlier versions
    scalefactor : float
      multiply all emissivities by this before deciding if they are
      above or below the broadening threshold
    temperature : float
      broaden all lines with emissivity*scalefactor > broadenlimit with
      Gaussians reflecting this electron temperature (Kelvin)
    broadenlimit : float
      lines with emissivity > this will be broadened.

    RETURNS
    -------
    spectrum : float(array)
      emissivity in photons s-1 bin-1. Note scalefactor is  *NOT* applied to this
    """

    if ((ebins_checksum == False ) | \
        (ebins_checksum != self.ebins_checksum)):
      # need to recalculate the continuum spectrum

      self.continuum = self.cont.return_spectrum(ebins) +\
                       self.pseudo.return_spectrum(ebins)


    # now do the lines!
    spec = numpy.zeros(len(ebins)-1)

    ### I HAVE NOT YET IMPLEMENTED BROADENING


#    if temperature == True:
#      # we are doing some broadening
#      emiss = self.linelist['epsilon'] * scalefactor
#      ibroad = emiss > broadenlimit
#      s,z = numpy.histogram(self.linelist['energy'][~ibroad], \
#                            bins=ebins, \
#                            weights = self.linelist['epsilon'][~ibroad])
#      spec += s
#
#      broadencoefft = temperature *pyatomdb.const.ERG_KEV/ pyatomdb.atomic.Z_to_mass(self.Z)* pyatomdb.const.AMUKG
#      for l in self.linelist[ibroad]:
#        # add each line with broadening


 #   else:
    s,z = numpy.histogram(self.linelist['energy'], \
                          bins=ebins, \
                          weights = self.linelist['epsilon'])
    spec += s


    spec += self.continuum

    return spec



class Continuum():
  """
  Class for holding a continuum for one ion

  """

  def __init__(self, E, C, conttype):
    """
    E : array(float)
    C : array(float)
    conttype : string
      "psuedo" or "continuum"
    """

    self.E = E
    self.C = C
    self.conttype = conttype



  def return_spectrum(self, ebins):
    """
    Return the spectrum of the ion on the grid ebins

    PARAMETERS
    ----------
    ebins : array(float)
      The bin edges, in keV

    RETURNS
    -------
    spectrum : array(float)
      The emissivity spectrum in ph cm^3 s^-1 bin^-1
    """

    #if ((not(ebins_checksum)) | (ebins_checksum != self.ebins_checksum)):
    spec = self.expand_spectrum(ebins)
    self.ebins_checksum = hashlib.md5(ebins).hexdigest()
    return spec




  def expand_spectrum(self, eedges):

    """
    Code to expand the compressed continuum onto a series of bins.

    Parameters
    ----------
    eedges : float(array)
      The bin edges for the spectrum to be calculated on, in units of keV

    Returns
    -------
    float(array)
      len(eedges)-1 array of continuum emission, in units of \
      photons cm^3 s^-1 bin^-1
    """
    import scipy.integrate
    n=len(self.E)
    if n==2:
      return 0.0
  # ok. So. Sort this.
    E_all = numpy.append(self.E, eedges)
    cont_tmp = numpy.interp(eedges, self.E, self.C)
    C_all = numpy.append(self.C, cont_tmp)

    iord = numpy.argsort(E_all)

  # order the arrays
    E_all = E_all[iord]
    C_all = C_all[iord]

    ihi = numpy.where(iord>=n)[0]
    cum_cont = scipy.integrate.cumtrapz(C_all, E_all, initial=0)
    C_out = numpy.zeros(len(eedges))
    C_out = cum_cont[ihi]

    cont = C_out[1:]-C_out[:-1]
    return cont


def modelabund(Z, abundset):
  modelabund={}
  modelabund['angr'] = numpy.zeros(31)
  modelabund['angr'][1]=1.00e+00
  modelabund['angr'][2]=9.77e-02
  modelabund['angr'][3]=1.45e-11
  modelabund['angr'][4]=1.41e-11
  modelabund['angr'][5]=3.98e-10
  modelabund['angr'][6]=3.63e-04
  modelabund['angr'][7]=1.12e-04
  modelabund['angr'][8]=8.51e-04
  modelabund['angr'][9]=3.63e-08
  modelabund['angr'][10]=1.23e-04
  modelabund['angr'][11]=2.14e-06
  modelabund['angr'][12]=3.80e-05
  modelabund['angr'][13]=2.95e-06
  modelabund['angr'][14]=3.55e-05
  modelabund['angr'][15]=2.82e-07
  modelabund['angr'][16]=1.62e-05
  modelabund['angr'][17]=3.16e-07
  modelabund['angr'][18]=3.63e-06
  modelabund['angr'][19]=1.32e-07
  modelabund['angr'][20]=2.29e-06
  modelabund['angr'][21]=1.26e-09
  modelabund['angr'][22]=9.77e-08
  modelabund['angr'][23]=1.00e-08
  modelabund['angr'][24]=4.68e-07
  modelabund['angr'][25]=2.45e-07
  modelabund['angr'][26]=4.68e-05
  modelabund['angr'][27]=8.32e-08
  modelabund['angr'][28]=1.78e-06
  modelabund['angr'][29]=1.62e-08
  modelabund['angr'][30]=3.98e-08

  modelabund['aspl'] = numpy.zeros(31)
  modelabund['aspl'][1]= 1.00e+00
  modelabund['aspl'][2]= 8.51e-02
  modelabund['aspl'][3]= 1.12e-11
  modelabund['aspl'][4]= 2.40e-11
  modelabund['aspl'][5]= 5.01e-10
  modelabund['aspl'][6]= 2.69e-04
  modelabund['aspl'][7]= 6.76e-05
  modelabund['aspl'][8]= 4.90e-04
  modelabund['aspl'][9]= 3.63e-08
  modelabund['aspl'][10]=8.51e-05
  modelabund['aspl'][11]=1.74e-06
  modelabund['aspl'][12]=3.98e-05
  modelabund['aspl'][13]=2.82e-06
  modelabund['aspl'][14]=3.24e-05
  modelabund['aspl'][15]=2.57e-07
  modelabund['aspl'][16]=1.32e-05
  modelabund['aspl'][17]=3.16e-07
  modelabund['aspl'][18]=2.51e-06
  modelabund['aspl'][19]=1.07e-07
  modelabund['aspl'][20]=2.19e-06
  modelabund['aspl'][21]=1.41e-09
  modelabund['aspl'][22]=8.91e-08
  modelabund['aspl'][23]=8.51e-09
  modelabund['aspl'][24]=4.37e-07
  modelabund['aspl'][25]=2.69e-07
  modelabund['aspl'][26]=3.16e-05
  modelabund['aspl'][27]=9.77e-08
  modelabund['aspl'][28]=1.66e-06
  modelabund['aspl'][29]=1.55e-08
  modelabund['aspl'][30]=3.63e-08

  modelabund['feld'] = numpy.zeros(31)
  modelabund['feld'][1]= 1.00e+00
  modelabund['feld'][2]= 9.77e-02
  modelabund['feld'][3]= 1.26e-11
  modelabund['feld'][4]= 2.51e-11
  modelabund['feld'][5]= 3.55e-10
  modelabund['feld'][6]= 3.98e-04
  modelabund['feld'][7]= 1.00e-04
  modelabund['feld'][8]= 8.51e-04
  modelabund['feld'][9]= 3.63e-08
  modelabund['feld'][10]=1.29e-04
  modelabund['feld'][11]=2.14e-06
  modelabund['feld'][12]=3.80e-05
  modelabund['feld'][13]=2.95e-06
  modelabund['feld'][14]=3.55e-05
  modelabund['feld'][15]=2.82e-07
  modelabund['feld'][16]=1.62e-05
  modelabund['feld'][17]=3.16e-07
  modelabund['feld'][18]=4.47e-06
  modelabund['feld'][19]=1.32e-07
  modelabund['feld'][20]=2.29e-06
  modelabund['feld'][21]=1.48e-09
  modelabund['feld'][22]=1.05e-07
  modelabund['feld'][23]=1.00e-08
  modelabund['feld'][24]=4.68e-07
  modelabund['feld'][25]=2.45e-07
  modelabund['feld'][26]=3.24e-05
  modelabund['feld'][27]=8.32e-08
  modelabund['feld'][28]=1.78e-06
  modelabund['feld'][29]=1.62e-08
  modelabund['feld'][30]=3.98e-08

  modelabund['aneb'] = numpy.zeros(31)
  modelabund['aneb'][1]= 1.00e+00
  modelabund['aneb'][2]= 8.01e-02
  modelabund['aneb'][3]= 2.19e-09
  modelabund['aneb'][4]= 2.87e-11
  modelabund['aneb'][5]= 8.82e-10
  modelabund['aneb'][6]= 4.45e-04
  modelabund['aneb'][7]= 9.12e-05
  modelabund['aneb'][8]= 7.39e-04
  modelabund['aneb'][9]= 3.10e-08
  modelabund['aneb'][10]=1.38e-04
  modelabund['aneb'][11]=2.10e-06
  modelabund['aneb'][12]=3.95e-05
  modelabund['aneb'][13]=3.12e-06
  modelabund['aneb'][14]=3.68e-05
  modelabund['aneb'][15]=3.82e-07
  modelabund['aneb'][16]=1.89e-05
  modelabund['aneb'][17]=1.93e-07
  modelabund['aneb'][18]=3.82e-06
  modelabund['aneb'][19]=1.39e-07
  modelabund['aneb'][20]=2.25e-06
  modelabund['aneb'][21]=1.24e-09
  modelabund['aneb'][22]=8.82e-08
  modelabund['aneb'][23]=1.08e-08
  modelabund['aneb'][24]=4.93e-07
  modelabund['aneb'][25]=3.50e-07
  modelabund['aneb'][26]=3.31e-05
  modelabund['aneb'][27]=8.27e-08
  modelabund['aneb'][28]=1.81e-06
  modelabund['aneb'][29]=1.89e-08
  modelabund['aneb'][30]=4.63e-08

  modelabund['grsa'] = numpy.zeros(31)
  modelabund['grsa'][1]= 1.00e+00
  modelabund['grsa'][2]= 8.51e-02
  modelabund['grsa'][3]= 1.26e-11
  modelabund['grsa'][4]= 2.51e-11
  modelabund['grsa'][5]= 3.55e-10
  modelabund['grsa'][6]= 3.31e-04
  modelabund['grsa'][7]= 8.32e-05
  modelabund['grsa'][8]= 6.76e-04
  modelabund['grsa'][9]= 3.63e-08
  modelabund['grsa'][10]=1.20e-04
  modelabund['grsa'][11]=2.14e-06
  modelabund['grsa'][12]=3.80e-05
  modelabund['grsa'][13]=2.95e-06
  modelabund['grsa'][14]=3.55e-05
  modelabund['grsa'][15]=2.82e-07
  modelabund['grsa'][16]=2.14e-05
  modelabund['grsa'][17]=3.16e-07
  modelabund['grsa'][18]=2.51e-06
  modelabund['grsa'][19]=1.32e-07
  modelabund['grsa'][20]=2.29e-06
  modelabund['grsa'][21]=1.48e-09
  modelabund['grsa'][22]=1.05e-07
  modelabund['grsa'][23]=1.00e-08
  modelabund['grsa'][24]=4.68e-07
  modelabund['grsa'][25]=2.45e-07
  modelabund['grsa'][26]=3.16e-05
  modelabund['grsa'][27]=8.32e-08
  modelabund['grsa'][28]=1.78e-06
  modelabund['grsa'][29]=1.62e-08
  modelabund['grsa'][30]=3.98e-08

  modelabund['wilm'] = numpy.zeros(31)
  modelabund['wilm'][1]= 1.00e+00
  modelabund['wilm'][2]= 9.77e-02
  modelabund['wilm'][3]= 0.00
  modelabund['wilm'][4]= 0.00
  modelabund['wilm'][5]= 0.00
  modelabund['wilm'][6]= 2.40e-04
  modelabund['wilm'][7]= 7.59e-05
  modelabund['wilm'][8]= 4.90e-04
  modelabund['wilm'][9]= 0.00
  modelabund['wilm'][10]=8.71e-05
  modelabund['wilm'][11]=1.45e-06
  modelabund['wilm'][12]=2.51e-05
  modelabund['wilm'][13]=2.14e-06
  modelabund['wilm'][14]=1.86e-05
  modelabund['wilm'][15]=2.63e-07
  modelabund['wilm'][16]=1.23e-05
  modelabund['wilm'][17]=1.32e-07
  modelabund['wilm'][18]=2.57e-06
  modelabund['wilm'][19]=0.00
  modelabund['wilm'][20]=1.58e-06
  modelabund['wilm'][21]=0.00
  modelabund['wilm'][22]=6.46e-08
  modelabund['wilm'][23]=0.00
  modelabund['wilm'][24]=3.24e-07
  modelabund['wilm'][25]=2.19e-07
  modelabund['wilm'][26]=2.69e-05
  modelabund['wilm'][27]=8.32e-08
  modelabund['wilm'][28]=1.12e-06
  modelabund['wilm'][29]=0.00
  modelabund['wilm'][30]=0.00

  modelabund['lodd'] = numpy.zeros(31)
  modelabund['lodd'][1]= 1.00e+00
  modelabund['lodd'][2]= 7.92e-02
  modelabund['lodd'][3]= 1.90e-09
  modelabund['lodd'][4]= 2.57e-11
  modelabund['lodd'][5]= 6.03e-10
  modelabund['lodd'][6]= 2.45e-04
  modelabund['lodd'][7]= 6.76e-05
  modelabund['lodd'][8]= 4.90e-04
  modelabund['lodd'][9]= 2.88e-08
  modelabund['lodd'][10]=7.41e-05
  modelabund['lodd'][11]=1.99e-06
  modelabund['lodd'][12]=3.55e-05
  modelabund['lodd'][13]=2.88e-06
  modelabund['lodd'][14]=3.47e-05
  modelabund['lodd'][15]=2.88e-07
  modelabund['lodd'][16]=1.55e-05
  modelabund['lodd'][17]=1.82e-07
  modelabund['lodd'][18]=3.55e-06
  modelabund['lodd'][19]=1.29e-07
  modelabund['lodd'][20]=2.19e-06
  modelabund['lodd'][21]=1.17e-09
  modelabund['lodd'][22]=8.32e-08
  modelabund['lodd'][23]=1.00e-08
  modelabund['lodd'][24]=4.47e-07
  modelabund['lodd'][25]=3.16e-07
  modelabund['lodd'][26]=2.95e-05
  modelabund['lodd'][27]=8.13e-08
  modelabund['lodd'][28]=1.66e-06
  modelabund['lodd'][29]=1.82e-08
  modelabund['lodd'][30]=4.27e-08

  return modelabund[abundset][Z]/modelabund['angr'][Z]
