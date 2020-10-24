package=libXext
$(package)_version=1.3.4
$(package)_download_path=http://xorg.freedesktop.org/releases/individual/lib/
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=59ad6fcce98deaecc14d39a672cf218ca37aba617c9a0f691cac3bcd28edf82b
$(package)_dependencies=xproto xextproto libX11 libXau

define $(package)_set_vars
  $(package)_config_opts=--disable-static
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
