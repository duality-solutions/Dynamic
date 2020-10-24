package=libSM
$(package)_version=1.2.3
$(package)_download_path=http://xorg.freedesktop.org/releases/individual/lib/
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=2d264499dcb05f56438dee12a1b4b71d76736ce7ba7aa6efbf15ebb113769cbb
$(package)_dependencies=xtrans xproto libICE

define $(package)_set_vars
  $(package)_config_opts=--without-libuuid  --without-xsltproc  --disable-docs --disable-static
  $(package)_config_opts_linux=--with-pic
endef

define $(package)_config_cmds
  $($(package)_autoconf)
endef

define $(package)_build_cmds
  $(MAKE)
endef

define $(package)_stage_cmds
  $(MAKE) DESTDIR=$($(package)_staging_dir) install
endef
