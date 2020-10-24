package=libX11
$(package)_version=1.6.12
$(package)_download_path=http://xorg.freedesktop.org/releases/individual/lib/
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=f108227469419ac04d196df0f3b80ce1f7f65059bb54c0de811f4d8e03fd6ec7
$(package)_dependencies=libxcb xtrans xextproto xproto

define $(package)_set_vars
$(package)_config_opts=--disable-xkb --disable-static
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
