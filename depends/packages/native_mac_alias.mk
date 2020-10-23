package=native_mac_alias
$(package)_version=2.1.0
$(package)_download_path=https://github.com/al45tair/mac_alias/archive/
$(package)_file_name=v$($(package)_version).tar.gz
$(package)_sha256_hash=69c9efa2086441c0f1880dd68f2fec6d94f0aa7579bf6296e728025ae216e66e
$(package)_install_libdir=$(build_prefix)/lib/python/dist-packages

define $(package)_build_cmds
    python setup.py build
endef

define $(package)_stage_cmds
    mkdir -p $($(package)_install_libdir) && \
    python setup.py install --root=$($(package)_staging_dir) --prefix=$(build_prefix) --install-lib=$($(package)_install_libdir)
endef
