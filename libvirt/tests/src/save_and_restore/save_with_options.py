import logging
import os
import re

from avocado.utils import process
from virttest import utils_misc
from virttest import utils_package
from virttest import virsh
from virttest.libvirt_xml import vm_xml
from virttest.utils_test import libvirt

from provider.save import save_base

LOG = logging.getLogger('avocado.test.' + __name__)
VIRSH_ARGS = {'debug': True, 'ignore_status': False}


def run(test, params, env):

    def check_output(pattern, output, cmd):
        msg = f'to find [{pattern}] in {cmd} output'
        LOG.debug(f'Expect {msg}')
        if not re.search(pattern, output):
            test.fail(f'Failed {msg}')

    vm_name = params.get('main_vm')
    vm = env.get_vm(vm_name)

    scenario = params.get('scenario', '')
    status_error = "yes" == params.get('status_error', 'no')
    error_msg = params.get('error_msg', '')
    expect_msg = params.get('expect_msg', '')
    timeout = params.get('timeout', 60)
    virsh_options = params.get('virsh_options', '')
    options = params.get('options', '')
    pre_state = params.get('pre_state', 'running')
    after_state = params.get('after_state', pre_state)
    rand_id = utils_misc.generate_random_string(3)
    save_path = f'/var/tmp/{vm_name}_{rand_id}.save'
    check_cmd = params.get('check_cmd', '')
    check_cmd = check_cmd.format(
        save_path, save_path) if check_cmd else check_cmd
    check_reason = 'yes' == params.get('check_reason', 'no')
    check_reason = check_reason and not status_error
    check_reason_cmd = params.get('check_reason_cmd')
    pattern_running = params.get('pattern_running')
    pattern_completed = params.get('pattern_completed')
    domst_reason = params.get('domst_reason')

    vmxml = vm_xml.VMXML.new_from_inactive_dumpxml(vm_name)
    bkxml = vmxml.copy()

    try:
        pid_ping, upsince = save_base.pre_save_setup(vm)
        if pre_state == 'paused':
            virsh.suspend(vm_name, **VIRSH_ARGS)

        if scenario == 'xml_opt':
            alter_xml = vm_xml.VMXML.new_from_dumpxml(vm_name,
                                                      options='--migratable')
            description = params.get('description', '{}')
            alter_xml.description = description
            params['vm_description'] = description
            LOG.debug(f"Modify description to {description}")
            LOG.debug(f'XML for saving:\n{alter_xml}')
            options += ' ' + alter_xml.xml

        if scenario == 'bypass_cache_opt':
            # Install lsof pkg if not installed
            if not utils_package.package_install("lsof"):
                test.cancel("Failed to install lsof in host\n")
            sp = process.SubProcess(check_cmd, shell=True)
            sp.start()

        if check_reason:
            monitor_sp = process.SubProcess(check_reason_cmd, shell=True)
            monitor_sp.start()

        save_result = virsh.save(vm_name, save_path, options=options,
                                 debug=True, virsh_opt=virsh_options)
        libvirt.check_exit_status(save_result, status_error)

        if check_reason:
            domjobinfo_output = monitor_sp.get_stdout().decode()
            monitor_sp.terminate()
            LOG.debug(f'domjobinfo output:\n{domjobinfo_output}')
            check_output(pattern_running, domjobinfo_output, 'domjobinfo')

            domjobinfo_completed = virsh.domjobinfo(
                vm_name, '--completed', **VIRSH_ARGS).stdout_text
            check_output(pattern_completed, domjobinfo_completed, 'domjobinfo')

        if status_error:
            libvirt.check_result(save_result, error_msg)
            if vm.state() != pre_state:
                test.fail(f'VM state should not changed since save failed.'
                          f'VM should be {pre_state}, not {vm.state()}')
            return
        else:
            if vm.state() != 'shut off':
                test.fail(f'VM should be shut off after being successfully '
                          f'saved, but current state is {vm.state()}')
            if expect_msg:
                save_output = save_result.stdout_text + save_result.stderr_text
                if not re.search(expect_msg, save_output):
                    test.fail(f'Expect content "{expect_msg}" not in output: '
                              f'{save_output}')

        if check_reason:
            domst_save = virsh.domstate(
                vm_name, '--reason', **VIRSH_ARGS).stdout_text
            pattern_save = 'shut off \(saved\)'
            check_output(pattern_save, domst_save, 'domstate')

        if scenario == 'bypass_cache_opt':
            output = sp.get_stdout().decode()
            LOG.debug(f'bypass-cache check output:\n{output}')
            sp.terminate()
            flags = os.O_DIRECT
            lines = re.findall(r"flags:.(\d+)", output, re.M)
            LOG.debug("Find all fdinfo flags: %s" % lines)
            lines = [int(i, 8) & flags for i in lines]
            if flags not in lines:
                test.fail('bypass-cache check fail, please check log')

        if scenario == 'xml_opt':
            vmxml_save = vm_xml.VMXML()
            vmxml_save.xml = virsh.save_image_dumpxml(save_path).stdout_text
            if vmxml_save.description != params['vm_description']:
                test.fail('VM description after save is incorrect')

        virsh.restore(save_path, **VIRSH_ARGS)

        if not utils_misc.wait_for(lambda: vm.state() == after_state,
                                   timeout=timeout):
            test.fail(f'VM should be {after_state} after restore, but current '
                      f'state is {vm.state()}')
        if check_reason:
            domst_restore = virsh.domstate(
                vm_name, '--reason', **VIRSH_ARGS).stdout_text
            pattern_restore = f'{pre_state}\s*\({domst_reason}\)'
            check_output(pattern_restore, domst_restore, 'domstate')

        if vm.state() == 'paused':
            virsh.resume(vm_name, **VIRSH_ARGS)

        save_base.post_save_check(vm, pid_ping, upsince)

        if scenario == 'xml_opt':
            vmxml_after = vm_xml.VMXML.new_from_dumpxml(vm_name)
            if vmxml_after.description != params['vm_description']:
                test.fail('VM description after save-restore is incorrect')

    finally:
        bkxml.sync()
        if os.path.exists(save_path):
            os.remove(save_path)
