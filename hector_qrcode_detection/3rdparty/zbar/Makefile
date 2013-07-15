all: zbar

INSTALL_DIR = .
TARBALL = build/zbar-0.10.tar.bz2
TARBALL_URL = http://sourceforge.net/projects/zbar/files/zbar/0.10/zbar-0.10.tar.bz2
SOURCE_DIR = build/zbar-0.10
MD5SUM_FILE = zbar-0.10.tar.bz2.md5sum
UNPACK_CMD = tar xjf
OPTIONS = --disable-video --without-imagemagick --without-qt --without-gtk --without-python --without-xshm --without-x --without-xv
include $(shell rospack find mk)/download_unpack_build.mk

zbar: $(SOURCE_DIR)/unpacked
	cd $(SOURCE_DIR) && ./configure --prefix=$(INSTALL_DIR) $(OPTIONS) && make install
	touch zbar
	
clean:
	-rm -rf $(SOURCE_DIR) build bin include lib share zbar

