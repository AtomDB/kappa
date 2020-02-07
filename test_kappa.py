import xspec, kappa2, pylab

import kappa_xspec
fig= pylab.figure()
fig.show()
ax = fig.add_subplot(111)

s1=xspec.Spectrum('hello.fak')



#fs1 = xspec.FakeitSettings(response='acisi_aimpt_cy22.rmf',\
#                           arf='acisi_aimpt_cy22.arf',\
#                           exposure = 1e6,\
#                           fileName = 'hello.fak')
xspec.Plot.xAxis='keV'

print("PING1")
m1 = xspec.Model('kappa')

xspec.Plot.device='/null'
#xspec.AllData.fakeit(1, [fs1])

xspec.Plot('data')

ax.plot(xspec.Plot.x(), xspec.Plot.model())


m2 = xspec.Model('vkappa')
xspec.Plot('data')

ax.plot(xspec.Plot.x(), xspec.Plot.model())


m3 = xspec.Model('vvkappa')
xspec.Plot('data')

ax.plot(xspec.Plot.x(), xspec.Plot.model())
pylab.draw()
zzz=input()
