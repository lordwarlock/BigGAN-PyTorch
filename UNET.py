from Pix2PixUNet import UnetGenerator
from BigGAN import Discriminator
from BigGAN import G_D
class Generator(UnetGenerator):
	def __init__(self, **kwargs):
		super(Generator, self).__init__(1, 3, 7)