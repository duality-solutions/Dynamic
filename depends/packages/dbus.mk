package=dbus
$(package)_version=1.13.4
$(package)_download_path=http://dbus.freedesktop.org/releases/dbus
$(package)_file_name=$(package)-$($(package)_version).tar.gz
$(package)_sha256_hash=8a8f0b986ac6214da9707da521bea9f49f09610083c71fdc8eddf8b4c54f384b
$(package)_dependencies=expat

define $(package)_set_vars
  $(package)_config_opts=--disable-tests --disable-doxygen-docs --disable-xml-docs --disable-static --without-x
endef

define $(package)_config_cmds
  $($(package)_autoconf)
endef

define $(package)_build_cmds
  $(MAKE) -C dbus libdbus-1.la
endef

define $(package)_stage_cmds
  $(MAKE) -C dbus DESTDIR=$($(package)_staging_dir) install-libLTLIBRARIES install-dbusincludeHEADERS install-nodist_dbusarchincludeHEADERS && \
  $(MAKE) DESTDIR=$($(package)_staging_dir) install-pkgconfigDATA
endef
