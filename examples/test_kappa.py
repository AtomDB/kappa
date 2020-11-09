# example for using kappa module in xspec

from kappa_xspec import *
import  pylab

# create a graph
fig= pylab.figure()
fig.show()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

# set xAxis to energy
xspec.Plot.xAxis='keV'

# declare the kappa model
m1 = xspec.Model('pykappa')

# turn off XSPEC loglog windows
xspec.Plot.device='/null'

m1.pykappa.kappa = 2.5
xspec.Plot('model')

ax1.loglog(xspec.Plot.x(), xspec.Plot.model(), label='kappa=2.5')

m1.pykappa.kappa = 25.0
xspec.Plot('model')
ax1.loglog(xspec.Plot.x(), xspec.Plot.model(), label='kappa=25')
ax1.set_xlim([0.3,10.0])

ax1.legend(loc=0)


#Now explore the variable kapp model, pyvkappa
m1 = xspec.Model('pyvkappa')
xspec.Plot('model')

ax2.loglog(xspec.Plot.x(), xspec.Plot.model(), label='all')

# turn off all the elements except iron

m1.pyvkappa.C = 0.0
m1.pyvkappa.N = 0.0
m1.pyvkappa.O = 0.0
m1.pyvkappa.Ne = 0.0
m1.pyvkappa.Mg = 0.0
m1.pyvkappa.Al = 0.0
m1.pyvkappa.Si = 0.0
m1.pyvkappa.S = 0.0
m1.pyvkappa.Ar = 0.0
m1.pyvkappa.Ca = 0.0
m1.pyvkappa.Ni = 0.0
m1.pyvkappa.Fe = 3.0
xspec.Plot('model')
ax2.loglog(xspec.Plot.x(), xspec.Plot.model(), label = 'Fe=3.0 only')
ax2.legend(loc=0)

ax2.set_xlabel('Energy (keV)')
ax1.set_ylabel('Ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$')
ax2.set_ylabel('Ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$')
pylab.draw()
zzz=input('Press any key to continue')

fig.savefig('test_kappa.pdf')
fig.savefig('test_kappa.svg')


