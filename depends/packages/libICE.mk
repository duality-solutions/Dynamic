package=libICE
$(package)_version=1.0.10
$(package)_download_path=http://xorg.freedesktop.org/releases/individual/lib/
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=6f86dce12cf4bcaf5c37dddd8b1b64ed2ddf1ef7b218f22b9942595fb747c348
$(package)_dependencies=xtrans xproto

define $(package)_set_vars
  $(package)_config_opts=--disable-static --disable-docs --disable-specs --without-xsltproc
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
