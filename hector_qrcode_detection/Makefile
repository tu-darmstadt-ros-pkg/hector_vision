include $(shell rospack find mk)/cmake.mk

clean: clean_zbar

clean_zbar:
	make -C 3rdparty/zbar clean
	-rm -rf include/zbar* lib/libzbar* lib share
