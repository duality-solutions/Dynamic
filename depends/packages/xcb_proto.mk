package=xcb_proto
$(package)_version=1.14
$(package)_download_path=http://xcb.freedesktop.org/dist
$(package)_file_name=xcb-proto-$($(package)_version).tar.gz
$(package)_sha256_hash=1c3fa23d091fb5e4f1e9bf145a902161cec00d260fabf880a7a248b02ab27031

define $(package)_set_vars
  $(package)_config_opts=--disable-shared
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

define $(package)_postprocess_cmds
  find -name "*.pyc" -delete && \
  find -name "*.pyo" -delete
endef
