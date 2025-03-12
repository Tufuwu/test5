import os
import logging as log
import random

from avocado.utils import process
from avocado.utils import cpu
from virttest import virsh
from virttest import libvirt_cgroup

SYSFS_SYSTEM_PATH = "/sys/devices/system/cpu"


# Using as lower capital is not the best way to do, but this is just a
# workaround to avoid changing the entire file.
logging = log.getLogger('avocado.' + __name__)


def get_present_cpu():
    """
    Get host present cpu

    :return: the host present cpu number
    """
    if os.path.exists("%s/cpu0" % SYSFS_SYSTEM_PATH):
        cmd = "ls %s | grep cpu[0-9] | wc -l" % SYSFS_SYSTEM_PATH
        cmd_result = process.run(cmd, ignore_status=True, shell=True)
        present = int(cmd_result.stdout_text.strip())
    else:
        present = None

    return present


def format_map(map_str, map_test, map_length):
    """
    Format cpu map str to tuple

    :param map_str: cpu map string
    :param map_test: template cpu map tuple
    :param map_length: cpu map tuple length
    :return: the cpu map tuple
    """
    cpu_map = ()
    if '-' in map_str:
        param = map_str.split('-')
        for i in range(map_length):
            if i in range(int(param[0]), int(param[1]) + 1):
                cpu_map += ('y',)
            else:
                cpu_map += (map_test[i],)
    else:
        for i in range(map_length):
            if i == int(map_str):
                cpu_map += ('y',)
            else:
                cpu_map += (map_test[i],)

    return cpu_map


def get_online_cpu(option=''):
    """
    Get host online cpu map and number

    :return: the host online cpu map tuple
    """
    cpu_map = ()
    map_test = ()
    cpu_map_list = []

    present = get_present_cpu()
    if not present:
        return None

    for i in range(present):
        map_test += ('-',)

    for i in range(present):
        if i == 0:
            cpu_map_list.append('y')
        else:
            cpu_map_list.append('-')

    if os.path.exists("%s/online" % SYSFS_SYSTEM_PATH):
        cmd = "cat %s/online" % SYSFS_SYSTEM_PATH
        cmd_result = process.run(cmd, ignore_status=True, shell=True)
        output = cmd_result.stdout_text.strip()
        if 'pretty' in option:
            return tuple(output)
        if ',' in output:
            output1 = output.split(',')
            for i in range(len(output1)):
                cpu_map = format_map(output1[i], map_test, present)
                map_test = cpu_map
        else:
            cpu_map = format_map(output, map_test, present)
    else:
        for i in range(present):
            if i != 0:
                if os.path.exists("%s/cpu%s/online" % (SYSFS_SYSTEM_PATH, i)):
                    cmd = "cat %s/cpu%s/online" % (SYSFS_SYSTEM_PATH, i)
                    cmd_result = process.run(cmd, ignore_status=True, shell=True)
                    output = cmd_result.stdout_text.strip()
                    if int(output) == 1:
                        cpu_map_list[i] = 'y'
                else:
                    return None
        cpu_map = tuple(cpu_map_list)

    return cpu_map


def check_result(result, option, status_error, test):
    """
    Check result of virsh nodecpumap
    :param result: The Cmd object of virsh nodecpumap
    :param option: The option of virsh nodecpumap
    :param status_error: Expect status error or not
    :param test: The test object
    :return: Success or raise Exception
    """

    output = result.stdout.strip()
    status = result.exit_status
    if status_error == "yes":
        if status == 0:
            test.fail("Run successfully with wrong command!")
        else:
            logging.info("Run failed as expected")
    else:
        out_value = []
        out = output.split('\n')
        for i in range(3):
            out_value.append(out[i].split()[-1])

        present = get_present_cpu()
        if not present:
            test.cancel("Host cpu counting not supported")
        else:
            if present != int(out_value[0]):
                test.fail("Present cpu is not expected")

        cpu_map = get_online_cpu(option)
        if not cpu_map:
            test.cancel("Host cpu map not supported")
        else:
            if cpu_map != tuple(out_value[2]):
                logging.info(cpu_map)
                logging.info(tuple(out_value[2]))
                test.fail("Cpu map is not expected")

        online = 0
        if 'pretty' in option:
            cpu_map = get_online_cpu()
        for i in range(present):
            if cpu_map[i] == 'y':
                online += 1
        if online != int(out_value[1]):
            test.fail("Online cpu is not expected")


def run(test, params, env):
    """
    Test the command virsh nodecpumap

    (1) Call virsh nodecpumap
    (2) Call virsh nodecpumap with pretty option
    (3) Call virsh nodecpumap with an unexpected option
    """

    option = params.get("virsh_node_options")
    status_error = params.get("status_error")
    cpu_off_on_test = params.get("cpu_off_on_test", "no") == "yes"
    online_cpus = cpu.cpu_online_list()
    test_cpu = random.choice(online_cpus)

    try:
        if cpu_off_on_test:
            # CPU offline will change default cpuset and this change will not
            # be reverted after re-online that cpu on v1 cgroup.
            # Need to revert cpuset manually on v1 cgroup.
            if not libvirt_cgroup.CgroupTest(None).is_cgroup_v2_enabled():
                logging.debug("Need to keep original value in cpuset file under "
                              "cgroup v1 environment for later recovery")
                default_cpuset = libvirt_cgroup.CgroupTest(None).get_cpuset_cpus(params.get('main_vm'))
            # Turn off CPU
            cpu.offline(test_cpu)

        result = virsh.nodecpumap(option, ignore_status=True, debug=True)
        check_result(result, option, status_error, test)

        if cpu_off_on_test:
            # Turn on CPU and check again
            cpu.online(test_cpu)
            result = virsh.nodecpumap(option, ignore_status=True, debug=True)
            check_result(result, option, status_error, test)
    finally:
        if cpu_off_on_test:
            if not libvirt_cgroup.CgroupTest(None).is_cgroup_v2_enabled():
                logging.debug("Reset cpuset file under cgroup v1 environment")
                try:
                    libvirt_cgroup.CgroupTest(None)\
                        .set_cpuset_cpus(default_cpuset, params.get('main_vm'))
                except Exception as e:
                    test.error("Revert cpuset failed: {}".format(str(e)))
