from forcegen import ForceGenerator

forceg = ForceGenerator(0.6, 0.0075, 0.025, 1, 0.05, 80)
forceg.shear_x(0,0,0) # shear op 0 0 0 in de x richting
forceg.show()