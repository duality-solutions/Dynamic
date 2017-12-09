package=qrencode
$(package)_version=4.0.0
$(package)_download_path=https://fukuchi.org/works/qrencode/
$(package)_file_name=qrencode-$(qrencode_version).tar.bz2
$(package)_sha256_hash=c90035e16921117d4086a7fdee65aab85be32beb4a376f6b664b8a425d327d0b

define $(package)_set_vars
$(package)_config_opts=--disable-shared -without-tools --disable-sdltest
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
