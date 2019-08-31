# -*- mode: ruby -*-
# vi: set ft=ruby :

VM_NAME = "dynamic-v2.4"
DYNAMIC_PATH = "/opt/dynamic"

# ugly hack to prevent hashicorp's bitrot. See https://github.com/hashicorp/vagrant/issues/9442
# this setting is required for pre-2.0 vagrant, but causes an error as of 2.0.3,
# remove entirely when confident nobody uses vagrant 1.x for anything.
unless Vagrant::DEFAULT_SERVER_URL.frozen?
  Vagrant::DEFAULT_SERVER_URL.replace('https://vagrantcloud.com')
end

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://vagrantcloud.com/search.
  # Base on the Sandstorm snapshots of the official Debian 9 (stretch) box with vboxsf support.
  config.vm.box = "ubuntu/bionic64"

  # vagrant plugin install vagrant-disksize
  if Vagrant.has_plugin?("vagrant-disksize") then
    config.disksize.size = "50GB"
  end


  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false

  if Vagrant.has_plugin?("vagrant-vbguest") then
    # vagrant-vbguest is a Vagrant plugin that upgrades
    # the version of VirtualBox Guest Additions within each
    # guest. If you have the vagrant-vbguest plugin, then it
    # needs to know how to compile kernel modules, etc., and so
    # we give it this hint about operating system type.
    config.vm.guest = "ubuntu"
  end

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine. In the example below,
  # accessing "localhost:8080" will access port 80 on the guest machine.
  # NOTE: This will enable public access to the opened port
  
  # Main net
  config.vm.network "forwarded_port", guest: 33300, host: 33300 # P2P
  config.vm.network "forwarded_port", guest: 33350, host: 33350, host_ip: "127.0.0.1" # RPC
  
  # Test net
  config.vm.network "forwarded_port", guest: 33400, host: 33400 # P2P
  config.vm.network "forwarded_port", guest: 33450, host: 33450, host_ip: "127.0.0.1" # RPC
  
  # ReqTest net
  config.vm.network "forwarded_port", guest: 33500, host: 33500 # P2P
  config.vm.network "forwarded_port", guest: 33550, host: 33550, host_ip: "127.0.0.1" # RPC

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  # config.vm.network "private_network", ip: "192.168.33.10"

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  # config.vm.network "public_network"

  # Calculate the number of CPUs and the amount of RAM the system has,
  # in a platform-dependent way; further logic below.
  cpus = nil
  total_kB_ram = nil

  host = RbConfig::CONFIG['host_os']
  if host =~ /darwin/
    cpus = `sysctl -n hw.ncpu`.to_i
    total_kB_ram =  `sysctl -n hw.memsize`.to_i / 1024
  elsif host =~ /linux/
    cpus = `nproc`.to_i
    total_kB_ram = `grep MemTotal /proc/meminfo | awk '{print $2}'`.to_i
  elsif host =~ /mingw/
    # powershell may not be available on Windows XP and Vista, so wrap this in a rescue block
    begin
      cpus = `powershell -Command "(Get-WmiObject Win32_Processor -Property NumberOfLogicalProcessors | Select-Object -Property NumberOfLogicalProcessors | Measure-Object NumberOfLogicalProcessors -Sum).Sum"`.to_i
      total_kB_ram = `powershell -Command "[math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory)"`.to_i / 1024
    rescue
    end
  end

  # Use the same number of CPUs within Vagrant as the system, with 1
  # as a default.
  #
  # Use at least 512MB of RAM, and if the system has more than 2GB of
  # RAM, use 1/4 of the system RAM. This seems a reasonable compromise
  # between having the Vagrant guest operating system not run out of
  # RAM entirely (which it basically would if we went much lower than
  # 512MB) and also allowing it to use up a healthily large amount of
  # RAM so it can run faster on systems that can afford it.
  if cpus.nil? or cpus.zero?
    cpus = 1
  end
  if total_kB_ram.nil? or total_kB_ram < 2048000
    assign_ram_mb = 512
  else
    assign_ram_mb = (total_kB_ram / 1024 / 4)
  end

  # Actually apply these CPU/memory values to the providers.
  config.vm.provider :virtualbox do |vb, override|
    vb.cpus = cpus
    vb.memory = assign_ram_mb
    vb.name = VM_NAME
    vb.customize ["modifyvm", :id, "--nictype1", "virtio"]

    # enables symlinks for windows
    override.vm.synced_folder ".", DYNAMIC_PATH
    override.vm.synced_folder ".dynamic", "/home/vagrant/.dynamic"
    vb.customize ["setextradata", :id, "VBoxInternal2/SharedFoldersEnableSymlinksCreate/#{DYNAMIC_PATH}", "1"]
  end

  config.vm.provider :libvirt do |libvirt, override|
    libvirt.cpus = cpus
    libvirt.memory = assign_ram_mb
    libvirt.default_prefix = VM_NAME

    # /opt/dynamic/dynamic and /root/.dynamic are used by vagrant-spk
    override.vm.synced_folder ".", DYNAMIC_PATH, type: "9p", accessmode: "passthrough"
    override.vm.synced_folder ".dynamic", "/home/vagrant/.dynamic", type: "9p", accessmode: "passthrough"
  end

  # View the documentation for the provider you are using for more
  # information on available options.

  # Enable provisioning with a shell script. Additional provisioners such as
  # Puppet, Chef, Ansible, Salt, and Docker are also available. Please see the
  # documentation for more information about their specific syntax and use.
  # config.vm.provision "shell", inline: "bash #{DYNAMIC_PATH}/dynamic-devenv/scripts/setup.sh", keep_color: true, env: {"VAGRANT" => "1"}
  config.vm.provision "shell", inline: <<-SHELL
    sudo add-apt-repository -y ppa:bitcoin/bitcoin
    sudo apt-get update
    sudo apt-get install -qy libdb4.8-dev libdb4.8++-dev build-essential libtool autotools-dev autoconf pkg-config libssl-dev libcrypto++-dev libevent-dev git libboost-all-dev libminiupnpc-dev libzmq3-dev
    sudo bash -c 'echo "Dynamic in /opt/dynamic" >/etc/motd'
    cd /opt/dynamic && ./autogen.sh && ./configure --without-gui --disable-gpu && make -j12
  SHELL

end
