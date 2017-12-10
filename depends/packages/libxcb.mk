package=libxcb
$(package)_version=1.12
$(package)_download_path=http://xcb.freedesktop.org/dist
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=4adfb1b7c67e99bc9c2ccb110b2f175686576d2f792c8a71b9c8b19014057b5b
$(package)_dependencies=xcb_proto libXau xproto

define $(package)_set_vars
$(package)_config_opts=--disable-static
endef

define $(package)_preprocess_cmds
  sed "s/pthread-stubs//" -i configure
endef

# Don't install xcb headers to the default path in order to work around a qt
# build issue: https://bugreports.qt.io/browse/QTBUG-34748
# When using qt's internal libxcb, it may end up finding the real headers in
# depends staging. Use a non-default path to avoid that.

define $(package)_config_cmds
  $($(package)_autoconf) --includedir=$(host_prefix)/include/xcb-shared
endef

define $(package)_build_cmds
  $(MAKE)
endef

define $(package)_stage_cmds
  $(MAKE) DESTDIR=$($(package)_staging_dir) install
endef

define $(package)_postprocess_cmds
  rm -rf share/man share/doc
endef
