package=expat
$(package)_version=2.2.10
$(package)_download_path=https://github.com/libexpat/libexpat/releases/download/R_2_2_10/
$(package)_file_name=$(package)-$($(package)_version).tar.bz2
$(package)_sha256_hash=b2c160f1b60e92da69de8e12333096aeb0c3bf692d41c60794de278af72135a5

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
