
driver:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64 
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 d9 3f 00 00 	mov    0x3fd9(%rip),%rax        # 4fe8 <__gmon_start__>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	callq  *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	retq   

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 6a 3f 00 00    	pushq  0x3f6a(%rip)        # 4f90 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	f2 ff 25 6b 3f 00 00 	bnd jmpq *0x3f6b(%rip)        # 4f98 <_GLOBAL_OFFSET_TABLE_+0x10>
    102d:	0f 1f 00             	nopl   (%rax)
    1030:	f3 0f 1e fa          	endbr64 
    1034:	68 00 00 00 00       	pushq  $0x0
    1039:	f2 e9 e1 ff ff ff    	bnd jmpq 1020 <.plt>
    103f:	90                   	nop
    1040:	f3 0f 1e fa          	endbr64 
    1044:	68 01 00 00 00       	pushq  $0x1
    1049:	f2 e9 d1 ff ff ff    	bnd jmpq 1020 <.plt>
    104f:	90                   	nop
    1050:	f3 0f 1e fa          	endbr64 
    1054:	68 02 00 00 00       	pushq  $0x2
    1059:	f2 e9 c1 ff ff ff    	bnd jmpq 1020 <.plt>
    105f:	90                   	nop
    1060:	f3 0f 1e fa          	endbr64 
    1064:	68 03 00 00 00       	pushq  $0x3
    1069:	f2 e9 b1 ff ff ff    	bnd jmpq 1020 <.plt>
    106f:	90                   	nop
    1070:	f3 0f 1e fa          	endbr64 
    1074:	68 04 00 00 00       	pushq  $0x4
    1079:	f2 e9 a1 ff ff ff    	bnd jmpq 1020 <.plt>
    107f:	90                   	nop
    1080:	f3 0f 1e fa          	endbr64 
    1084:	68 05 00 00 00       	pushq  $0x5
    1089:	f2 e9 91 ff ff ff    	bnd jmpq 1020 <.plt>
    108f:	90                   	nop
    1090:	f3 0f 1e fa          	endbr64 
    1094:	68 06 00 00 00       	pushq  $0x6
    1099:	f2 e9 81 ff ff ff    	bnd jmpq 1020 <.plt>
    109f:	90                   	nop

Disassembly of section .plt.got:

00000000000010a0 <__cxa_finalize@plt>:
    10a0:	f3 0f 1e fa          	endbr64 
    10a4:	f2 ff 25 4d 3f 00 00 	bnd jmpq *0x3f4d(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    10ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000010b0 <free@plt>:
    10b0:	f3 0f 1e fa          	endbr64 
    10b4:	f2 ff 25 e5 3e 00 00 	bnd jmpq *0x3ee5(%rip)        # 4fa0 <free@GLIBC_2.2.5>
    10bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010c0 <__stack_chk_fail@plt>:
    10c0:	f3 0f 1e fa          	endbr64 
    10c4:	f2 ff 25 dd 3e 00 00 	bnd jmpq *0x3edd(%rip)        # 4fa8 <__stack_chk_fail@GLIBC_2.4>
    10cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010d0 <calloc@plt>:
    10d0:	f3 0f 1e fa          	endbr64 
    10d4:	f2 ff 25 d5 3e 00 00 	bnd jmpq *0x3ed5(%rip)        # 4fb0 <calloc@GLIBC_2.2.5>
    10db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010e0 <memmove@plt>:
    10e0:	f3 0f 1e fa          	endbr64 
    10e4:	f2 ff 25 cd 3e 00 00 	bnd jmpq *0x3ecd(%rip)        # 4fb8 <memmove@GLIBC_2.2.5>
    10eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010f0 <__fprintf_chk@plt>:
    10f0:	f3 0f 1e fa          	endbr64 
    10f4:	f2 ff 25 c5 3e 00 00 	bnd jmpq *0x3ec5(%rip)        # 4fc0 <__fprintf_chk@GLIBC_2.3.4>
    10fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001100 <rand@plt>:
    1100:	f3 0f 1e fa          	endbr64 
    1104:	f2 ff 25 bd 3e 00 00 	bnd jmpq *0x3ebd(%rip)        # 4fc8 <rand@GLIBC_2.2.5>
    110b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001110 <__sprintf_chk@plt>:
    1110:	f3 0f 1e fa          	endbr64 
    1114:	f2 ff 25 b5 3e 00 00 	bnd jmpq *0x3eb5(%rip)        # 4fd0 <__sprintf_chk@GLIBC_2.3.4>
    111b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000001140 <main>:
    1140:	f3 0f 1e fa          	endbr64 
    1144:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    1149:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
    114d:	41 ff 72 f8          	pushq  -0x8(%r10)
    1151:	55                   	push   %rbp
    1152:	48 89 e5             	mov    %rsp,%rbp
    1155:	41 57                	push   %r15
    1157:	41 56                	push   %r14
    1159:	41 55                	push   %r13
    115b:	41 54                	push   %r12
    115d:	41 52                	push   %r10
    115f:	53                   	push   %rbx
    1160:	48 81 ec 80 01 00 00 	sub    $0x180,%rsp
    1167:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    116e:	00 00 
    1170:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    1174:	31 c0                	xor    %eax,%eax
    1176:	83 ff 01             	cmp    $0x1,%edi
    1179:	7e 44                	jle    11bf <main+0x7f>
    117b:	48 8b 35 9e 3e 00 00 	mov    0x3e9e(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1182:	48 8d bd 70 fe ff ff 	lea    -0x190(%rbp),%rdi
    1189:	48 8d 15 78 1e 00 00 	lea    0x1e78(%rip),%rdx        # 3008 <_IO_stdin_used+0x8>
    1190:	e8 8b 08 00 00       	callq  1a20 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>
    1195:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    1199:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    11a0:	00 00 
    11a2:	0f 85 ff 01 00 00    	jne    13a7 <main+0x267>
    11a8:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    11ac:	5b                   	pop    %rbx
    11ad:	41 5a                	pop    %r10
    11af:	41 5c                	pop    %r12
    11b1:	41 5d                	pop    %r13
    11b3:	41 5e                	pop    %r14
    11b5:	41 5f                	pop    %r15
    11b7:	31 c0                	xor    %eax,%eax
    11b9:	5d                   	pop    %rbp
    11ba:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    11be:	c3                   	retq   
    11bf:	31 c0                	xor    %eax,%eax
    11c1:	48 8d bd 50 ff ff ff 	lea    -0xb0(%rbp),%rdi
    11c8:	b9 1c 00 00 00       	mov    $0x1c,%ecx
    11cd:	f3 ab                	rep stos %eax,%es:(%rdi)
    11cf:	52                   	push   %rdx
    11d0:	48 8d 0d 6b 1e 00 00 	lea    0x1e6b(%rip),%rcx        # 3042 <_IO_stdin_used+0x42>
    11d7:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
    11de:	6a 02                	pushq  $0x2
    11e0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    11e4:	48 89 c7             	mov    %rax,%rdi
    11e7:	6a 01                	pushq  $0x1
    11e9:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
    11f0:	41 b9 11 00 00 00    	mov    $0x11,%r9d
    11f6:	51                   	push   %rcx
    11f7:	4c 8d 05 2d 1e 00 00 	lea    0x1e2d(%rip),%r8        # 302b <_IO_stdin_used+0x2b>
    11fe:	48 8d 0d 2e 1e 00 00 	lea    0x1e2e(%rip),%rcx        # 3033 <_IO_stdin_used+0x33>
    1205:	ba 80 00 00 00       	mov    $0x80,%edx
    120a:	be 01 00 00 00       	mov    $0x1,%esi
    120f:	31 c0                	xor    %eax,%eax
    1211:	c5 f8 29 85 40 ff ff 	vmovaps %xmm0,-0xc0(%rbp)
    1218:	ff 
    1219:	e8 f2 fe ff ff       	callq  1110 <__sprintf_chk@plt>
    121e:	48 83 c4 20          	add    $0x20,%rsp
    1222:	be 08 00 00 00       	mov    $0x8,%esi
    1227:	bf 00 01 00 00       	mov    $0x100,%edi
    122c:	e8 9f fe ff ff       	callq  10d0 <calloc@plt>
    1231:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
    1238:	41 bd 02 00 00 00    	mov    $0x2,%r13d
    123e:	4c 8d a5 d0 fe ff ff 	lea    -0x130(%rbp),%r12
    1245:	bb 00 00 10 00       	mov    $0x100000,%ebx
    124a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1250:	e8 ab fe ff ff       	callq  1100 <rand@plt>
    1255:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    125b:	e8 a0 fe ff ff       	callq  1100 <rand@plt>
    1260:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    1266:	e8 95 fe ff ff       	callq  1100 <rand@plt>
    126b:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    1271:	e8 8a fe ff ff       	callq  1100 <rand@plt>
    1276:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    127c:	e8 7f fe ff ff       	callq  1100 <rand@plt>
    1281:	89 85 e0 fe ff ff    	mov    %eax,-0x120(%rbp)
    1287:	0f 31                	rdtsc  
    1289:	4c 89 e7             	mov    %r12,%rdi
    128c:	e8 af 04 00 00       	callq  1740 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE2EEvPT0_>
    1291:	0f 31                	rdtsc  
    1293:	ff cb                	dec    %ebx
    1295:	75 b9                	jne    1250 <main+0x110>
    1297:	41 83 fd 01          	cmp    $0x1,%r13d
    129b:	0f 85 fb 00 00 00    	jne    139c <main+0x25c>
    12a1:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
    12a8:	49 89 c7             	mov    %rax,%r15
    12ab:	48 05 00 08 00 00    	add    $0x800,%rax
    12b1:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
    12b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    12bf:	00 
    12c0:	41 be 00 00 10 00    	mov    $0x100000,%r14d
    12c6:	31 db                	xor    %ebx,%ebx
    12c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    12cf:	00 
    12d0:	e8 2b fe ff ff       	callq  1100 <rand@plt>
    12d5:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    12db:	e8 20 fe ff ff       	callq  1100 <rand@plt>
    12e0:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    12e6:	e8 15 fe ff ff       	callq  1100 <rand@plt>
    12eb:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    12f1:	e8 0a fe ff ff       	callq  1100 <rand@plt>
    12f6:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    12fc:	e8 ff fd ff ff       	callq  1100 <rand@plt>
    1301:	89 85 e0 fe ff ff    	mov    %eax,-0x120(%rbp)
    1307:	0f 31                	rdtsc  
    1309:	49 89 c5             	mov    %rax,%r13
    130c:	48 c1 e2 20          	shl    $0x20,%rdx
    1310:	4c 89 e7             	mov    %r12,%rdi
    1313:	49 09 d5             	or     %rdx,%r13
    1316:	e8 25 04 00 00       	callq  1740 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE2EEvPT0_>
    131b:	0f 31                	rdtsc  
    131d:	48 c1 e2 20          	shl    $0x20,%rdx
    1321:	48 09 d0             	or     %rdx,%rax
    1324:	4c 29 e8             	sub    %r13,%rax
    1327:	48 01 c3             	add    %rax,%rbx
    132a:	41 ff ce             	dec    %r14d
    132d:	75 a1                	jne    12d0 <main+0x190>
    132f:	62 f1 ff 08 7b c3    	vcvtusi2sd %rbx,%xmm0,%xmm0
    1335:	49 83 c7 08          	add    $0x8,%r15
    1339:	c5 fb 59 05 df 20 00 	vmulsd 0x20df(%rip),%xmm0,%xmm0        # 3420 <_IO_stdin_used+0x420>
    1340:	00 
    1341:	c4 c1 7b 11 47 f8    	vmovsd %xmm0,-0x8(%r15)
    1347:	4c 3b bd 68 fe ff ff 	cmp    -0x198(%rbp),%r15
    134e:	0f 85 6c ff ff ff    	jne    12c0 <main+0x180>
    1354:	48 8b 9d 60 fe ff ff 	mov    -0x1a0(%rbp),%rbx
    135b:	4c 8d a5 70 fe ff ff 	lea    -0x190(%rbp),%r12
    1362:	31 c9                	xor    %ecx,%ecx
    1364:	48 89 de             	mov    %rbx,%rsi
    1367:	4c 89 e7             	mov    %r12,%rdi
    136a:	ba 00 01 00 00       	mov    $0x100,%edx
    136f:	e8 0c 0c 00 00       	callq  1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    1374:	48 8b 8d 58 fe ff ff 	mov    -0x1a8(%rbp),%rcx
    137b:	48 8b 35 9e 3c 00 00 	mov    0x3c9e(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1382:	4c 89 e7             	mov    %r12,%rdi
    1385:	ba 01 00 00 00       	mov    $0x1,%edx
    138a:	e8 d1 06 00 00       	callq  1a60 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>
    138f:	48 89 df             	mov    %rbx,%rdi
    1392:	e8 19 fd ff ff       	callq  10b0 <free@plt>
    1397:	e9 f9 fd ff ff       	jmpq   1195 <main+0x55>
    139c:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    13a2:	e9 9e fe ff ff       	jmpq   1245 <main+0x105>
    13a7:	e8 14 fd ff ff       	callq  10c0 <__stack_chk_fail@plt>
    13ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000013b0 <_start>:
    13b0:	f3 0f 1e fa          	endbr64 
    13b4:	31 ed                	xor    %ebp,%ebp
    13b6:	49 89 d1             	mov    %rdx,%r9
    13b9:	5e                   	pop    %rsi
    13ba:	48 89 e2             	mov    %rsp,%rdx
    13bd:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    13c1:	50                   	push   %rax
    13c2:	54                   	push   %rsp
    13c3:	4c 8d 05 56 11 00 00 	lea    0x1156(%rip),%r8        # 2520 <__libc_csu_fini>
    13ca:	48 8d 0d df 10 00 00 	lea    0x10df(%rip),%rcx        # 24b0 <__libc_csu_init>
    13d1:	48 8d 3d 68 fd ff ff 	lea    -0x298(%rip),%rdi        # 1140 <main>
    13d8:	ff 15 02 3c 00 00    	callq  *0x3c02(%rip)        # 4fe0 <__libc_start_main@GLIBC_2.2.5>
    13de:	f4                   	hlt    
    13df:	90                   	nop

00000000000013e0 <deregister_tm_clones>:
    13e0:	48 8d 3d 29 3c 00 00 	lea    0x3c29(%rip),%rdi        # 5010 <__TMC_END__>
    13e7:	48 8d 05 22 3c 00 00 	lea    0x3c22(%rip),%rax        # 5010 <__TMC_END__>
    13ee:	48 39 f8             	cmp    %rdi,%rax
    13f1:	74 15                	je     1408 <deregister_tm_clones+0x28>
    13f3:	48 8b 05 de 3b 00 00 	mov    0x3bde(%rip),%rax        # 4fd8 <_ITM_deregisterTMCloneTable>
    13fa:	48 85 c0             	test   %rax,%rax
    13fd:	74 09                	je     1408 <deregister_tm_clones+0x28>
    13ff:	ff e0                	jmpq   *%rax
    1401:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1408:	c3                   	retq   
    1409:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001410 <register_tm_clones>:
    1410:	48 8d 3d f9 3b 00 00 	lea    0x3bf9(%rip),%rdi        # 5010 <__TMC_END__>
    1417:	48 8d 35 f2 3b 00 00 	lea    0x3bf2(%rip),%rsi        # 5010 <__TMC_END__>
    141e:	48 29 fe             	sub    %rdi,%rsi
    1421:	48 89 f0             	mov    %rsi,%rax
    1424:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1428:	48 c1 f8 03          	sar    $0x3,%rax
    142c:	48 01 c6             	add    %rax,%rsi
    142f:	48 d1 fe             	sar    %rsi
    1432:	74 14                	je     1448 <register_tm_clones+0x38>
    1434:	48 8b 05 b5 3b 00 00 	mov    0x3bb5(%rip),%rax        # 4ff0 <_ITM_registerTMCloneTable>
    143b:	48 85 c0             	test   %rax,%rax
    143e:	74 08                	je     1448 <register_tm_clones+0x38>
    1440:	ff e0                	jmpq   *%rax
    1442:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1448:	c3                   	retq   
    1449:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001450 <__do_global_dtors_aux>:
    1450:	f3 0f 1e fa          	endbr64 
    1454:	80 3d cd 3b 00 00 00 	cmpb   $0x0,0x3bcd(%rip)        # 5028 <completed.8060>
    145b:	75 2b                	jne    1488 <__do_global_dtors_aux+0x38>
    145d:	55                   	push   %rbp
    145e:	48 83 3d 92 3b 00 00 	cmpq   $0x0,0x3b92(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    1465:	00 
    1466:	48 89 e5             	mov    %rsp,%rbp
    1469:	74 0c                	je     1477 <__do_global_dtors_aux+0x27>
    146b:	48 8b 3d 96 3b 00 00 	mov    0x3b96(%rip),%rdi        # 5008 <__dso_handle>
    1472:	e8 29 fc ff ff       	callq  10a0 <__cxa_finalize@plt>
    1477:	e8 64 ff ff ff       	callq  13e0 <deregister_tm_clones>
    147c:	c6 05 a5 3b 00 00 01 	movb   $0x1,0x3ba5(%rip)        # 5028 <completed.8060>
    1483:	5d                   	pop    %rbp
    1484:	c3                   	retq   
    1485:	0f 1f 00             	nopl   (%rax)
    1488:	c3                   	retq   
    1489:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001490 <frame_dummy>:
    1490:	f3 0f 1e fa          	endbr64 
    1494:	e9 77 ff ff ff       	jmpq   1410 <register_tm_clones>
    1499:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000014a0 <_ZN5vsort5sortaIhLj17ENS_7bitonicELNS_17simd_instructionsE1ELNS_13builtin_usageE2EEEvPT_>:
    14a0:	f3 0f 1e fa          	endbr64 
    14a4:	62 f1 fd 28 6f 0f    	vmovdqa64 (%rdi),%ymm1
    14aa:	62 f1 fd 28 6f 1d 4c 	vmovdqa64 0x1d4c(%rip),%ymm3        # 3200 <_IO_stdin_used+0x200>
    14b1:	1d 00 00 
    14b4:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    14ba:	c4 e2 75 00 05 9d 1b 	vpshufb 0x1b9d(%rip),%ymm1,%ymm0        # 3060 <_IO_stdin_used+0x60>
    14c1:	00 00 
    14c3:	c4 e2 6d 00 15 b4 1b 	vpshufb 0x1bb4(%rip),%ymm2,%ymm2        # 3080 <_IO_stdin_used+0x80>
    14ca:	00 00 
    14cc:	c5 fd eb d2          	vpor   %ymm2,%ymm0,%ymm2
    14d0:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    14d4:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    14d8:	62 f1 fd 28 6f 0d be 	vmovdqa64 0x1bbe(%rip),%ymm1        # 30a0 <_IO_stdin_used+0xa0>
    14df:	1b 00 00 
    14e2:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    14e8:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    14ee:	c4 e2 75 00 05 c9 1b 	vpshufb 0x1bc9(%rip),%ymm1,%ymm0        # 30c0 <_IO_stdin_used+0xc0>
    14f5:	00 00 
    14f7:	c4 e2 6d 00 15 e0 1b 	vpshufb 0x1be0(%rip),%ymm2,%ymm2        # 30e0 <_IO_stdin_used+0xe0>
    14fe:	00 00 
    1500:	c5 fd eb c2          	vpor   %ymm2,%ymm0,%ymm0
    1504:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    1508:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    150c:	62 f1 fd 28 6f 05 ea 	vmovdqa64 0x1bea(%rip),%ymm0        # 3100 <_IO_stdin_used+0x100>
    1513:	1b 00 00 
    1516:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    151c:	c4 e3 fd 00 c8 4e    	vpermq $0x4e,%ymm0,%ymm1
    1522:	c4 e2 7d 00 15 f5 1b 	vpshufb 0x1bf5(%rip),%ymm0,%ymm2        # 3120 <_IO_stdin_used+0x120>
    1529:	00 00 
    152b:	c4 e2 75 00 0d 0c 1c 	vpshufb 0x1c0c(%rip),%ymm1,%ymm1        # 3140 <_IO_stdin_used+0x140>
    1532:	00 00 
    1534:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
    1538:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    153c:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    1540:	62 f1 fd 28 6f 15 16 	vmovdqa64 0x1c16(%rip),%ymm2        # 3160 <_IO_stdin_used+0x160>
    1547:	1c 00 00 
    154a:	c4 e3 7d 4c d1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm2
    1550:	c4 e3 fd 00 c2 4e    	vpermq $0x4e,%ymm2,%ymm0
    1556:	c4 e2 6d 00 0d 21 1c 	vpshufb 0x1c21(%rip),%ymm2,%ymm1        # 3180 <_IO_stdin_used+0x180>
    155d:	00 00 
    155f:	c4 e2 7d 00 05 38 1c 	vpshufb 0x1c38(%rip),%ymm0,%ymm0        # 31a0 <_IO_stdin_used+0x1a0>
    1566:	00 00 
    1568:	c5 f5 eb c8          	vpor   %ymm0,%ymm1,%ymm1
    156c:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    1570:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    1574:	62 f1 fd 28 6f 0d 42 	vmovdqa64 0x1c42(%rip),%ymm1        # 31c0 <_IO_stdin_used+0x1c0>
    157b:	1c 00 00 
    157e:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    1584:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    158a:	c4 e2 75 00 05 4d 1c 	vpshufb 0x1c4d(%rip),%ymm1,%ymm0        # 31e0 <_IO_stdin_used+0x1e0>
    1591:	00 00 
    1593:	c4 e2 6d 00 d3       	vpshufb %ymm3,%ymm2,%ymm2
    1598:	c5 fd eb c2          	vpor   %ymm2,%ymm0,%ymm0
    159c:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    15a0:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    15a4:	62 f1 fd 28 6f 05 72 	vmovdqa64 0x1c72(%rip),%ymm0        # 3220 <_IO_stdin_used+0x220>
    15ab:	1c 00 00 
    15ae:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    15b4:	c4 e3 fd 00 c8 4e    	vpermq $0x4e,%ymm0,%ymm1
    15ba:	c4 e2 7d 00 15 7d 1c 	vpshufb 0x1c7d(%rip),%ymm0,%ymm2        # 3240 <_IO_stdin_used+0x240>
    15c1:	00 00 
    15c3:	c4 e2 75 00 cb       	vpshufb %ymm3,%ymm1,%ymm1
    15c8:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
    15cc:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    15d0:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    15d4:	62 f1 fd 28 6f 15 82 	vmovdqa64 0x1c82(%rip),%ymm2        # 3260 <_IO_stdin_used+0x260>
    15db:	1c 00 00 
    15de:	c4 e3 7d 4c d1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm2
    15e4:	c4 e3 fd 00 c2 4e    	vpermq $0x4e,%ymm2,%ymm0
    15ea:	c4 e2 6d 00 0d 8d 1c 	vpshufb 0x1c8d(%rip),%ymm2,%ymm1        # 3280 <_IO_stdin_used+0x280>
    15f1:	00 00 
    15f3:	c4 e2 7d 00 05 a4 1c 	vpshufb 0x1ca4(%rip),%ymm0,%ymm0        # 32a0 <_IO_stdin_used+0x2a0>
    15fa:	00 00 
    15fc:	c5 f5 eb c8          	vpor   %ymm0,%ymm1,%ymm1
    1600:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    1604:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    1608:	62 f1 fd 28 6f 0d ae 	vmovdqa64 0x1cae(%rip),%ymm1        # 32c0 <_IO_stdin_used+0x2c0>
    160f:	1c 00 00 
    1612:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    1618:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    161e:	c4 e2 75 00 05 b9 1c 	vpshufb 0x1cb9(%rip),%ymm1,%ymm0        # 32e0 <_IO_stdin_used+0x2e0>
    1625:	00 00 
    1627:	c4 e2 6d 00 d3       	vpshufb %ymm3,%ymm2,%ymm2
    162c:	c5 fd eb c2          	vpor   %ymm2,%ymm0,%ymm0
    1630:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    1634:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    1638:	62 f1 fd 28 6f 05 be 	vmovdqa64 0x1cbe(%rip),%ymm0        # 3300 <_IO_stdin_used+0x300>
    163f:	1c 00 00 
    1642:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    1648:	c4 e3 fd 00 c8 4e    	vpermq $0x4e,%ymm0,%ymm1
    164e:	c4 e2 7d 00 15 c9 1c 	vpshufb 0x1cc9(%rip),%ymm0,%ymm2        # 3320 <_IO_stdin_used+0x320>
    1655:	00 00 
    1657:	c4 e2 75 00 cb       	vpshufb %ymm3,%ymm1,%ymm1
    165c:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
    1660:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    1664:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    1668:	62 f1 fd 28 6f 15 ce 	vmovdqa64 0x1cce(%rip),%ymm2        # 3340 <_IO_stdin_used+0x340>
    166f:	1c 00 00 
    1672:	c4 e3 7d 4c d1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm2
    1678:	c4 e3 fd 00 c2 4e    	vpermq $0x4e,%ymm2,%ymm0
    167e:	c4 e2 6d 00 0d d9 1c 	vpshufb 0x1cd9(%rip),%ymm2,%ymm1        # 3360 <_IO_stdin_used+0x360>
    1685:	00 00 
    1687:	c4 e2 7d 00 c3       	vpshufb %ymm3,%ymm0,%ymm0
    168c:	c5 f5 eb c8          	vpor   %ymm0,%ymm1,%ymm1
    1690:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    1694:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    1698:	62 f1 fd 28 6f 0d de 	vmovdqa64 0x1cde(%rip),%ymm1        # 3380 <_IO_stdin_used+0x380>
    169f:	1c 00 00 
    16a2:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    16a8:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    16ae:	c4 e2 75 00 05 e9 1c 	vpshufb 0x1ce9(%rip),%ymm1,%ymm0        # 33a0 <_IO_stdin_used+0x3a0>
    16b5:	00 00 
    16b7:	c4 e2 6d 00 d3       	vpshufb %ymm3,%ymm2,%ymm2
    16bc:	c5 fd eb c2          	vpor   %ymm2,%ymm0,%ymm0
    16c0:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    16c4:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    16c8:	62 f1 fd 28 6f 05 ee 	vmovdqa64 0x1cee(%rip),%ymm0        # 33c0 <_IO_stdin_used+0x3c0>
    16cf:	1c 00 00 
    16d2:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    16d8:	c4 e3 fd 00 d0 4e    	vpermq $0x4e,%ymm0,%ymm2
    16de:	c4 e2 7d 00 0d f9 1c 	vpshufb 0x1cf9(%rip),%ymm0,%ymm1        # 33e0 <_IO_stdin_used+0x3e0>
    16e5:	00 00 
    16e7:	c4 e2 6d 00 db       	vpshufb %ymm3,%ymm2,%ymm3
    16ec:	c5 f5 eb db          	vpor   %ymm3,%ymm1,%ymm3
    16f0:	62 f1 fd 28 6f 15 06 	vmovdqa64 0x1d06(%rip),%ymm2        # 3400 <_IO_stdin_used+0x400>
    16f7:	1d 00 00 
    16fa:	c5 fd da cb          	vpminub %ymm3,%ymm0,%ymm1
    16fe:	c5 fd de c3          	vpmaxub %ymm3,%ymm0,%ymm0
    1702:	c4 e3 7d 4c c1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm0
    1708:	62 f1 fd 28 7f 07    	vmovdqa64 %ymm0,(%rdi)
    170e:	c5 f8 77             	vzeroupper 
    1711:	c3                   	retq   
    1712:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1719:	00 00 00 
    171c:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1723:	00 00 00 
    1726:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    172d:	00 00 00 
    1730:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1737:	00 00 00 
    173a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001740 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE2EEvPT0_>:
    1740:	f3 0f 1e fa          	endbr64 
    1744:	e9 57 fd ff ff       	jmpq   14a0 <_ZN5vsort5sortaIhLj17ENS_7bitonicELNS_17simd_instructionsE1ELNS_13builtin_usageE2EEEvPT_>
    1749:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001750 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>:
    1750:	f3 0f 1e fa          	endbr64 
    1754:	41 56                	push   %r14
    1756:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    175a:	31 d2                	xor    %edx,%edx
    175c:	41 55                	push   %r13
    175e:	41 54                	push   %r12
    1760:	55                   	push   %rbp
    1761:	48 89 f5             	mov    %rsi,%rbp
    1764:	53                   	push   %rbx
    1765:	48 89 fb             	mov    %rdi,%rbx
    1768:	48 81 ec 10 02 00 00 	sub    $0x210,%rsp
    176f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1776:	00 00 
    1778:	48 89 84 24 08 02 00 	mov    %rax,0x208(%rsp)
    177f:	00 
    1780:	31 c0                	xor    %eax,%eax
    1782:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
    1787:	c5 f8 29 04 24       	vmovaps %xmm0,(%rsp)
    178c:	89 c1                	mov    %eax,%ecx
    178e:	83 c0 20             	add    $0x20,%eax
    1791:	48 89 54 0c 10       	mov    %rdx,0x10(%rsp,%rcx,1)
    1796:	48 89 54 0c 18       	mov    %rdx,0x18(%rsp,%rcx,1)
    179b:	48 89 54 0c 20       	mov    %rdx,0x20(%rsp,%rcx,1)
    17a0:	48 89 54 0c 28       	mov    %rdx,0x28(%rsp,%rcx,1)
    17a5:	3d e0 01 00 00       	cmp    $0x1e0,%eax
    17aa:	72 e0                	jb     178c <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x3c>
    17ac:	c5 fb 10 05 b4 1e 00 	vmovsd 0x1eb4(%rip),%xmm0        # 3668 <_ZN6timersL12units_to_strE+0x28>
    17b3:	00 
    17b4:	c5 fb 10 4b 48       	vmovsd 0x48(%rbx),%xmm1
    17b9:	8b 53 50             	mov    0x50(%rbx),%edx
    17bc:	4c 8d 0d 5d 1e 00 00 	lea    0x1e5d(%rip),%r9        # 3620 <_ZN5statsL13possible_fmtsE>
    17c3:	48 01 c6             	add    %rax,%rsi
    17c6:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17ca:	48 8d 05 6f 1e 00 00 	lea    0x1e6f(%rip),%rax        # 3640 <_ZN6timersL12units_to_strE>
    17d1:	48 8d 04 d0          	lea    (%rax,%rdx,8),%rax
    17d5:	c5 fb 10 4b 40       	vmovsd 0x40(%rbx),%xmm1
    17da:	49 8d 51 08          	lea    0x8(%r9),%rdx
    17de:	49 89 d6             	mov    %rdx,%r14
    17e1:	4d 0f 43 f1          	cmovae %r9,%r14
    17e5:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17e9:	c5 fb 10 4b 38       	vmovsd 0x38(%rbx),%xmm1
    17ee:	49 89 d5             	mov    %rdx,%r13
    17f1:	4d 0f 43 e9          	cmovae %r9,%r13
    17f5:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17f9:	c5 fb 10 4b 30       	vmovsd 0x30(%rbx),%xmm1
    17fe:	49 89 d3             	mov    %rdx,%r11
    1801:	4d 0f 43 d9          	cmovae %r9,%r11
    1805:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1809:	c5 fb 10 4b 28       	vmovsd 0x28(%rbx),%xmm1
    180e:	49 89 d2             	mov    %rdx,%r10
    1811:	4d 0f 43 d1          	cmovae %r9,%r10
    1815:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1819:	c5 fb 10 4b 20       	vmovsd 0x20(%rbx),%xmm1
    181e:	49 89 d0             	mov    %rdx,%r8
    1821:	4d 0f 43 c1          	cmovae %r9,%r8
    1825:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1829:	c5 fb 10 4b 18       	vmovsd 0x18(%rbx),%xmm1
    182e:	48 89 d7             	mov    %rdx,%rdi
    1831:	49 0f 43 f9          	cmovae %r9,%rdi
    1835:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1839:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    183e:	48 c7 06 00 00 00 00 	movq   $0x0,(%rsi)
    1845:	48 c7 46 08 00 00 00 	movq   $0x0,0x8(%rsi)
    184c:	00 
    184d:	48 89 d6             	mov    %rdx,%rsi
    1850:	49 0f 43 f1          	cmovae %r9,%rsi
    1854:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1858:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    185d:	48 89 d1             	mov    %rdx,%rcx
    1860:	49 0f 43 c9          	cmovae %r9,%rcx
    1864:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1868:	49 89 e4             	mov    %rsp,%r12
    186b:	4c 0f 42 ca          	cmovb  %rdx,%r9
    186f:	48 83 ec 08          	sub    $0x8,%rsp
    1873:	50                   	push   %rax
    1874:	ba 00 02 00 00       	mov    $0x200,%edx
    1879:	41 56                	push   %r14
    187b:	50                   	push   %rax
    187c:	41 55                	push   %r13
    187e:	50                   	push   %rax
    187f:	41 53                	push   %r11
    1881:	50                   	push   %rax
    1882:	41 52                	push   %r10
    1884:	50                   	push   %rax
    1885:	41 50                	push   %r8
    1887:	4c 8b 03             	mov    (%rbx),%r8
    188a:	50                   	push   %rax
    188b:	57                   	push   %rdi
    188c:	4c 89 e7             	mov    %r12,%rdi
    188f:	50                   	push   %rax
    1890:	56                   	push   %rsi
    1891:	be 01 00 00 00       	mov    $0x1,%esi
    1896:	50                   	push   %rax
    1897:	51                   	push   %rcx
    1898:	48 8d 0d 89 1b 00 00 	lea    0x1b89(%rip),%rcx        # 3428 <_IO_stdin_used+0x428>
    189f:	50                   	push   %rax
    18a0:	31 c0                	xor    %eax,%eax
    18a2:	e8 69 f8 ff ff       	callq  1110 <__sprintf_chk@plt>
    18a7:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    18ae:	ff 73 48             	pushq  0x48(%rbx)
    18b1:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    18b6:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    18bb:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    18c0:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    18c5:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    18ca:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    18cf:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    18d4:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    18d9:	4c 89 e2             	mov    %r12,%rdx
    18dc:	be 01 00 00 00       	mov    $0x1,%esi
    18e1:	48 89 ef             	mov    %rbp,%rdi
    18e4:	b8 08 00 00 00       	mov    $0x8,%eax
    18e9:	e8 02 f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    18ee:	58                   	pop    %rax
    18ef:	5a                   	pop    %rdx
    18f0:	48 8b 84 24 08 02 00 	mov    0x208(%rsp),%rax
    18f7:	00 
    18f8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    18ff:	00 00 
    1901:	75 10                	jne    1913 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x1c3>
    1903:	48 81 c4 10 02 00 00 	add    $0x210,%rsp
    190a:	5b                   	pop    %rbx
    190b:	5d                   	pop    %rbp
    190c:	41 5c                	pop    %r12
    190e:	41 5d                	pop    %r13
    1910:	41 5e                	pop    %r14
    1912:	c3                   	retq   
    1913:	e8 a8 f7 ff ff       	callq  10c0 <__stack_chk_fail@plt>
    1918:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    191f:	00 

0000000000001920 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_>:
    1920:	f3 0f 1e fa          	endbr64 
    1924:	41 55                	push   %r13
    1926:	4d 89 c5             	mov    %r8,%r13
    1929:	41 54                	push   %r12
    192b:	49 89 d4             	mov    %rdx,%r12
    192e:	48 89 ca             	mov    %rcx,%rdx
    1931:	55                   	push   %rbp
    1932:	48 89 fd             	mov    %rdi,%rbp
    1935:	40 84 f6             	test   %sil,%sil
    1938:	74 16                	je     1950 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x30>
    193a:	5d                   	pop    %rbp
    193b:	4c 89 e6             	mov    %r12,%rsi
    193e:	41 5c                	pop    %r12
    1940:	4c 89 c1             	mov    %r8,%rcx
    1943:	41 5d                	pop    %r13
    1945:	e9 06 fe ff ff       	jmpq   1750 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>
    194a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1950:	48 83 f9 01          	cmp    $0x1,%rcx
    1954:	76 09                	jbe    195f <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x3f>
    1956:	4d 85 c0             	test   %r8,%r8
    1959:	0f 85 81 00 00 00    	jne    19e0 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xc0>
    195f:	48 83 fa 01          	cmp    $0x1,%rdx
    1963:	0f 85 8d 00 00 00    	jne    19f6 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xd6>
    1969:	4d 85 ed             	test   %r13,%r13
    196c:	74 19                	je     1987 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x67>
    196e:	4c 89 e9             	mov    %r13,%rcx
    1971:	48 8d 15 8c 1c 00 00 	lea    0x1c8c(%rip),%rdx        # 3604 <_IO_stdin_used+0x604>
    1978:	be 01 00 00 00       	mov    $0x1,%esi
    197d:	4c 89 e7             	mov    %r12,%rdi
    1980:	31 c0                	xor    %eax,%eax
    1982:	e8 69 f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1987:	48 83 ec 08          	sub    $0x8,%rsp
    198b:	ff 75 48             	pushq  0x48(%rbp)
    198e:	c5 fb 10 45 08       	vmovsd 0x8(%rbp),%xmm0
    1993:	c5 fb 10 7d 40       	vmovsd 0x40(%rbp),%xmm7
    1998:	c5 fb 10 75 38       	vmovsd 0x38(%rbp),%xmm6
    199d:	c5 fb 10 6d 30       	vmovsd 0x30(%rbp),%xmm5
    19a2:	c5 fb 10 65 28       	vmovsd 0x28(%rbp),%xmm4
    19a7:	c5 fb 10 5d 20       	vmovsd 0x20(%rbp),%xmm3
    19ac:	c5 fb 10 55 18       	vmovsd 0x18(%rbp),%xmm2
    19b1:	c5 fb 10 4d 10       	vmovsd 0x10(%rbp),%xmm1
    19b6:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
    19ba:	48 8d 15 af 1b 00 00 	lea    0x1baf(%rip),%rdx        # 3570 <_IO_stdin_used+0x570>
    19c1:	4c 89 e7             	mov    %r12,%rdi
    19c4:	be 01 00 00 00       	mov    $0x1,%esi
    19c9:	b8 08 00 00 00       	mov    $0x8,%eax
    19ce:	e8 1d f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    19d3:	58                   	pop    %rax
    19d4:	5a                   	pop    %rdx
    19d5:	5d                   	pop    %rbp
    19d6:	41 5c                	pop    %r12
    19d8:	41 5d                	pop    %r13
    19da:	c3                   	retq   
    19db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    19e0:	48 8d 15 1d 1c 00 00 	lea    0x1c1d(%rip),%rdx        # 3604 <_IO_stdin_used+0x604>
    19e7:	be 01 00 00 00       	mov    $0x1,%esi
    19ec:	4c 89 e7             	mov    %r12,%rdi
    19ef:	31 c0                	xor    %eax,%eax
    19f1:	e8 fa f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    19f6:	48 8d 15 3b 1b 00 00 	lea    0x1b3b(%rip),%rdx        # 3538 <_IO_stdin_used+0x538>
    19fd:	be 01 00 00 00       	mov    $0x1,%esi
    1a02:	4c 89 e7             	mov    %r12,%rdi
    1a05:	31 c0                	xor    %eax,%eax
    1a07:	e8 e4 f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1a0c:	e9 58 ff ff ff       	jmpq   1969 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x49>
    1a11:	90                   	nop
    1a12:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1a19:	00 00 00 00 
    1a1d:	0f 1f 00             	nopl   (%rax)

0000000000001a20 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>:
    1a20:	f3 0f 1e fa          	endbr64 
    1a24:	55                   	push   %rbp
    1a25:	48 89 f5             	mov    %rsi,%rbp
    1a28:	48 85 d2             	test   %rdx,%rdx
    1a2b:	74 19                	je     1a46 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc+0x26>
    1a2d:	48 89 d1             	mov    %rdx,%rcx
    1a30:	be 01 00 00 00       	mov    $0x1,%esi
    1a35:	48 8d 15 c8 1b 00 00 	lea    0x1bc8(%rip),%rdx        # 3604 <_IO_stdin_used+0x604>
    1a3c:	48 89 ef             	mov    %rbp,%rdi
    1a3f:	31 c0                	xor    %eax,%eax
    1a41:	e8 aa f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1a46:	48 89 ef             	mov    %rbp,%rdi
    1a49:	48 8d 15 e8 1a 00 00 	lea    0x1ae8(%rip),%rdx        # 3538 <_IO_stdin_used+0x538>
    1a50:	be 01 00 00 00       	mov    $0x1,%esi
    1a55:	31 c0                	xor    %eax,%eax
    1a57:	5d                   	pop    %rbp
    1a58:	e9 93 f6 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    1a5d:	90                   	nop
    1a5e:	66 90                	xchg   %ax,%ax

0000000000001a60 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>:
    1a60:	f3 0f 1e fa          	endbr64 
    1a64:	41 54                	push   %r12
    1a66:	49 89 cc             	mov    %rcx,%r12
    1a69:	55                   	push   %rbp
    1a6a:	48 89 f5             	mov    %rsi,%rbp
    1a6d:	53                   	push   %rbx
    1a6e:	48 89 fb             	mov    %rdi,%rbx
    1a71:	48 83 fa 01          	cmp    $0x1,%rdx
    1a75:	76 09                	jbe    1a80 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x20>
    1a77:	48 85 c9             	test   %rcx,%rcx
    1a7a:	0f 85 80 00 00 00    	jne    1b00 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xa0>
    1a80:	48 83 fa 01          	cmp    $0x1,%rdx
    1a84:	0f 85 8f 00 00 00    	jne    1b19 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xb9>
    1a8a:	4d 85 e4             	test   %r12,%r12
    1a8d:	74 19                	je     1aa8 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x48>
    1a8f:	4c 89 e1             	mov    %r12,%rcx
    1a92:	48 8d 15 6b 1b 00 00 	lea    0x1b6b(%rip),%rdx        # 3604 <_IO_stdin_used+0x604>
    1a99:	be 01 00 00 00       	mov    $0x1,%esi
    1a9e:	48 89 ef             	mov    %rbp,%rdi
    1aa1:	31 c0                	xor    %eax,%eax
    1aa3:	e8 48 f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1aa8:	48 83 ec 08          	sub    $0x8,%rsp
    1aac:	ff 73 48             	pushq  0x48(%rbx)
    1aaf:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    1ab4:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    1ab9:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    1abe:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    1ac3:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    1ac8:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    1acd:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    1ad2:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    1ad7:	48 8b 0b             	mov    (%rbx),%rcx
    1ada:	48 8d 15 8f 1a 00 00 	lea    0x1a8f(%rip),%rdx        # 3570 <_IO_stdin_used+0x570>
    1ae1:	48 89 ef             	mov    %rbp,%rdi
    1ae4:	be 01 00 00 00       	mov    $0x1,%esi
    1ae9:	b8 08 00 00 00       	mov    $0x8,%eax
    1aee:	e8 fd f5 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1af3:	58                   	pop    %rax
    1af4:	5a                   	pop    %rdx
    1af5:	5b                   	pop    %rbx
    1af6:	5d                   	pop    %rbp
    1af7:	41 5c                	pop    %r12
    1af9:	c3                   	retq   
    1afa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1b00:	48 89 d1             	mov    %rdx,%rcx
    1b03:	be 01 00 00 00       	mov    $0x1,%esi
    1b08:	48 8d 15 f5 1a 00 00 	lea    0x1af5(%rip),%rdx        # 3604 <_IO_stdin_used+0x604>
    1b0f:	48 89 ef             	mov    %rbp,%rdi
    1b12:	31 c0                	xor    %eax,%eax
    1b14:	e8 d7 f5 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1b19:	48 8d 15 18 1a 00 00 	lea    0x1a18(%rip),%rdx        # 3538 <_IO_stdin_used+0x538>
    1b20:	be 01 00 00 00       	mov    $0x1,%esi
    1b25:	48 89 ef             	mov    %rbp,%rdi
    1b28:	31 c0                	xor    %eax,%eax
    1b2a:	e8 c1 f5 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1b2f:	e9 56 ff ff ff       	jmpq   1a8a <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x2a>
    1b34:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1b3b:	00 00 00 00 
    1b3f:	90                   	nop

0000000000001b40 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>:
    1b40:	f3 0f 1e fa          	endbr64 
    1b44:	89 4f 50             	mov    %ecx,0x50(%rdi)
    1b47:	83 fa 01             	cmp    $0x1,%edx
    1b4a:	0f 84 d0 01 00 00    	je     1d20 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1e0>
    1b50:	89 d0                	mov    %edx,%eax
    1b52:	48 89 07             	mov    %rax,(%rdi)
    1b55:	89 d0                	mov    %edx,%eax
    1b57:	d1 e8                	shr    %eax
    1b59:	89 c1                	mov    %eax,%ecx
    1b5b:	c5 fb 10 04 ce       	vmovsd (%rsi,%rcx,8),%xmm0
    1b60:	f6 c2 01             	test   $0x1,%dl
    1b63:	0f 85 97 01 00 00    	jne    1d00 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1c0>
    1b69:	ff c0                	inc    %eax
    1b6b:	c5 fb 58 04 c6       	vaddsd (%rsi,%rax,8),%xmm0,%xmm0
    1b70:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    1b77:	00 
    1b78:	c5 fb 59 05 f0 1a 00 	vmulsd 0x1af0(%rip),%xmm0,%xmm0        # 3670 <_ZN6timersL12units_to_strE+0x30>
    1b7f:	00 
    1b80:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1b85:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    1b89:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    1b8d:	85 d2                	test   %edx,%edx
    1b8f:	74 20                	je     1bb1 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x71>
    1b91:	31 c0                	xor    %eax,%eax
    1b93:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    1b97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    1b9e:	00 00 
    1ba0:	c5 eb 58 14 c6       	vaddsd (%rsi,%rax,8),%xmm2,%xmm2
    1ba5:	48 ff c0             	inc    %rax
    1ba8:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    1bad:	39 c2                	cmp    %eax,%edx
    1baf:	77 ef                	ja     1ba0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x60>
    1bb1:	62 f1 57 08 7b ea    	vcvtusi2sd %edx,%xmm5,%xmm5
    1bb7:	8d 42 ff             	lea    -0x1(%rdx),%eax
    1bba:	48 89 c1             	mov    %rax,%rcx
    1bbd:	c5 eb 5e d5          	vdivsd %xmm5,%xmm2,%xmm2
    1bc1:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    1bc6:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1bcb:	c5 fb 11 4f 18       	vmovsd %xmm1,0x18(%rdi)
    1bd0:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1bd4:	c5 fb 11 4f 20       	vmovsd %xmm1,0x20(%rdi)
    1bd9:	c5 d3 59 0d 97 1a 00 	vmulsd 0x1a97(%rip),%xmm5,%xmm1        # 3678 <_ZN6timersL12units_to_strE+0x38>
    1be0:	00 
    1be1:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1be7:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1bec:	c5 fb 11 4f 28       	vmovsd %xmm1,0x28(%rdi)
    1bf1:	c5 d3 59 0d 87 1a 00 	vmulsd 0x1a87(%rip),%xmm5,%xmm1        # 3680 <_ZN6timersL12units_to_strE+0x40>
    1bf8:	00 
    1bf9:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1bff:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1c04:	c5 fb 11 4f 30       	vmovsd %xmm1,0x30(%rdi)
    1c09:	c5 d3 59 0d 77 1a 00 	vmulsd 0x1a77(%rip),%xmm5,%xmm1        # 3688 <_ZN6timersL12units_to_strE+0x48>
    1c10:	00 
    1c11:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1c17:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1c1c:	c5 fb 11 4f 38       	vmovsd %xmm1,0x38(%rdi)
    1c21:	83 fa 01             	cmp    $0x1,%edx
    1c24:	0f 86 56 01 00 00    	jbe    1d80 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x240>
    1c2a:	83 f9 02             	cmp    $0x2,%ecx
    1c2d:	0f 86 67 01 00 00    	jbe    1d9a <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x25a>
    1c33:	41 89 d0             	mov    %edx,%r8d
    1c36:	41 c1 e8 02          	shr    $0x2,%r8d
    1c3a:	41 ff c8             	dec    %r8d
    1c3d:	49 c1 e0 05          	shl    $0x5,%r8
    1c41:	c4 e2 7d 19 f2       	vbroadcastsd %xmm2,%ymm6
    1c46:	48 89 f0             	mov    %rsi,%rax
    1c49:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    1c4e:	c5 f9 28 d8          	vmovapd %xmm0,%xmm3
    1c52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1c58:	c5 fd 10 38          	vmovupd (%rax),%ymm7
    1c5c:	48 83 c0 20          	add    $0x20,%rax
    1c60:	c5 c5 5c c6          	vsubpd %ymm6,%ymm7,%ymm0
    1c64:	c5 fd 59 c0          	vmulpd %ymm0,%ymm0,%ymm0
    1c68:	c5 fb 58 db          	vaddsd %xmm3,%xmm0,%xmm3
    1c6c:	c5 f9 15 c8          	vunpckhpd %xmm0,%xmm0,%xmm1
    1c70:	62 f3 fd 28 19 c0 01 	vextractf64x2 $0x1,%ymm0,%xmm0
    1c77:	c5 f3 58 cb          	vaddsd %xmm3,%xmm1,%xmm1
    1c7b:	c5 f3 58 c8          	vaddsd %xmm0,%xmm1,%xmm1
    1c7f:	c5 f9 15 c0          	vunpckhpd %xmm0,%xmm0,%xmm0
    1c83:	c5 f3 58 d8          	vaddsd %xmm0,%xmm1,%xmm3
    1c87:	4c 39 c0             	cmp    %r8,%rax
    1c8a:	75 cc                	jne    1c58 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x118>
    1c8c:	89 d0                	mov    %edx,%eax
    1c8e:	83 e0 fc             	and    $0xfffffffc,%eax
    1c91:	f6 c2 03             	test   $0x3,%dl
    1c94:	0f 84 d6 00 00 00    	je     1d70 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1c9a:	41 89 c0             	mov    %eax,%r8d
    1c9d:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1ca3:	ff c0                	inc    %eax
    1ca5:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1ca9:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1cae:	39 c2                	cmp    %eax,%edx
    1cb0:	0f 86 ba 00 00 00    	jbe    1d70 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1cb6:	c5 f8 77             	vzeroupper 
    1cb9:	41 89 c0             	mov    %eax,%r8d
    1cbc:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1cc2:	ff c0                	inc    %eax
    1cc4:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1cc8:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1ccd:	39 c2                	cmp    %eax,%edx
    1ccf:	76 0e                	jbe    1cdf <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1cd1:	c5 fb 10 04 c6       	vmovsd (%rsi,%rax,8),%xmm0
    1cd6:	c5 fb 5c d2          	vsubsd %xmm2,%xmm0,%xmm2
    1cda:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
    1cdf:	62 f1 7f 08 7b c1    	vcvtusi2sd %ecx,%xmm0,%xmm0
    1ce5:	c5 e3 5e c0          	vdivsd %xmm0,%xmm3,%xmm0
    1ce9:	c5 fb 11 47 40       	vmovsd %xmm0,0x40(%rdi)
    1cee:	c5 e3 5e c5          	vdivsd %xmm5,%xmm3,%xmm0
    1cf2:	c5 fb 11 47 48       	vmovsd %xmm0,0x48(%rdi)
    1cf7:	c3                   	retq   
    1cf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1cff:	00 
    1d00:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1d05:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    1d0c:	00 
    1d0d:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    1d11:	e9 7b fe ff ff       	jmpq   1b91 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x51>
    1d16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1d1d:	00 00 00 
    1d20:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1d24:	48 c7 07 01 00 00 00 	movq   $0x1,(%rdi)
    1d2b:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1d30:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1d34:	c5 fb 11 47 10       	vmovsd %xmm0,0x10(%rdi)
    1d39:	c5 fb 11 47 18       	vmovsd %xmm0,0x18(%rdi)
    1d3e:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1d42:	c5 fb 11 47 20       	vmovsd %xmm0,0x20(%rdi)
    1d47:	c5 fb 11 47 28       	vmovsd %xmm0,0x28(%rdi)
    1d4c:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1d50:	48 c7 47 40 00 00 00 	movq   $0x0,0x40(%rdi)
    1d57:	00 
    1d58:	48 c7 47 48 00 00 00 	movq   $0x0,0x48(%rdi)
    1d5f:	00 
    1d60:	c5 fb 11 47 30       	vmovsd %xmm0,0x30(%rdi)
    1d65:	c5 fb 11 47 38       	vmovsd %xmm0,0x38(%rdi)
    1d6a:	c3                   	retq   
    1d6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1d70:	c5 f8 77             	vzeroupper 
    1d73:	e9 67 ff ff ff       	jmpq   1cdf <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1d78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1d7f:	00 
    1d80:	48 8b 3d 99 32 00 00 	mov    0x3299(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    1d87:	48 8d 15 12 18 00 00 	lea    0x1812(%rip),%rdx        # 35a0 <_IO_stdin_used+0x5a0>
    1d8e:	be 01 00 00 00       	mov    $0x1,%esi
    1d93:	31 c0                	xor    %eax,%eax
    1d95:	e9 56 f3 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    1d9a:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1d9e:	b8 01 00 00 00       	mov    $0x1,%eax
    1da3:	c5 f3 5c ca          	vsubsd %xmm2,%xmm1,%xmm1
    1da7:	c5 f9 28 d9          	vmovapd %xmm1,%xmm3
    1dab:	c4 e2 f9 99 d9       	vfmadd132sd %xmm1,%xmm0,%xmm3
    1db0:	e9 04 ff ff ff       	jmpq   1cb9 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x179>
    1db5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1dbc:	00 00 00 00 

0000000000001dc0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>:
    1dc0:	48 89 f0             	mov    %rsi,%rax
    1dc3:	48 29 f8             	sub    %rdi,%rax
    1dc6:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1dcc:	0f 8e a6 01 00 00    	jle    1f78 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1b8>
    1dd2:	41 56                	push   %r14
    1dd4:	49 89 d6             	mov    %rdx,%r14
    1dd7:	41 55                	push   %r13
    1dd9:	4c 8d 6f 08          	lea    0x8(%rdi),%r13
    1ddd:	41 54                	push   %r12
    1ddf:	55                   	push   %rbp
    1de0:	53                   	push   %rbx
    1de1:	48 89 fb             	mov    %rdi,%rbx
    1de4:	48 85 d2             	test   %rdx,%rdx
    1de7:	0f 84 03 01 00 00    	je     1ef0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x130>
    1ded:	48 c1 f8 04          	sar    $0x4,%rax
    1df1:	48 8d 04 c3          	lea    (%rbx,%rax,8),%rax
    1df5:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    1dfa:	c5 fb 10 10          	vmovsd (%rax),%xmm2
    1dfe:	49 ff ce             	dec    %r14
    1e01:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    1e05:	c5 fb 10 5e f8       	vmovsd -0x8(%rsi),%xmm3
    1e0a:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1e0e:	76 70                	jbe    1e80 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xc0>
    1e10:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1e14:	0f 87 b6 00 00 00    	ja     1ed0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1e1a:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1e1e:	77 6c                	ja     1e8c <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xcc>
    1e20:	c5 f1 14 c0          	vunpcklpd %xmm0,%xmm1,%xmm0
    1e24:	c5 f8 11 03          	vmovups %xmm0,(%rbx)
    1e28:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1e2d:	4c 89 ed             	mov    %r13,%rbp
    1e30:	48 89 f0             	mov    %rsi,%rax
    1e33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1e38:	c5 fb 10 55 00       	vmovsd 0x0(%rbp),%xmm2
    1e3d:	49 89 ec             	mov    %rbp,%r12
    1e40:	c5 f9 2f ca          	vcomisd %xmm2,%xmm1
    1e44:	77 30                	ja     1e76 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xb6>
    1e46:	48 83 e8 08          	sub    $0x8,%rax
    1e4a:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1e4e:	76 0f                	jbe    1e5f <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x9f>
    1e50:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1e55:	48 83 e8 08          	sub    $0x8,%rax
    1e59:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1e5d:	77 f1                	ja     1e50 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x90>
    1e5f:	48 39 e8             	cmp    %rbp,%rax
    1e62:	76 3c                	jbe    1ea0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xe0>
    1e64:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1e69:	c5 fb 11 10          	vmovsd %xmm2,(%rax)
    1e6d:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1e71:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1e76:	48 83 c5 08          	add    $0x8,%rbp
    1e7a:	eb bc                	jmp    1e38 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x78>
    1e7c:	0f 1f 40 00          	nopl   0x0(%rax)
    1e80:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1e84:	77 9a                	ja     1e20 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x60>
    1e86:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1e8a:	76 44                	jbe    1ed0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1e8c:	c5 fb 11 1b          	vmovsd %xmm3,(%rbx)
    1e90:	c5 fb 11 46 f8       	vmovsd %xmm0,-0x8(%rsi)
    1e95:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1e99:	eb 92                	jmp    1e2d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1e9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1ea0:	4c 89 f2             	mov    %r14,%rdx
    1ea3:	48 89 ef             	mov    %rbp,%rdi
    1ea6:	e8 15 ff ff ff       	callq  1dc0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1eab:	48 89 e8             	mov    %rbp,%rax
    1eae:	48 29 d8             	sub    %rbx,%rax
    1eb1:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1eb7:	0f 8e ac 00 00 00    	jle    1f69 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1a9>
    1ebd:	4d 85 f6             	test   %r14,%r14
    1ec0:	74 31                	je     1ef3 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x133>
    1ec2:	48 89 ee             	mov    %rbp,%rsi
    1ec5:	e9 23 ff ff ff       	jmpq   1ded <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x2d>
    1eca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1ed0:	c5 fb 11 13          	vmovsd %xmm2,(%rbx)
    1ed4:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    1ed8:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1edc:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1ee1:	e9 47 ff ff ff       	jmpq   1e2d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1ee6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1eed:	00 00 00 
    1ef0:	49 89 f4             	mov    %rsi,%r12
    1ef3:	48 c1 f8 03          	sar    $0x3,%rax
    1ef7:	4c 8d 68 fe          	lea    -0x2(%rax),%r13
    1efb:	49 d1 fd             	sar    %r13
    1efe:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1f04:	48 89 c2             	mov    %rax,%rdx
    1f07:	4c 89 ee             	mov    %r13,%rsi
    1f0a:	48 89 df             	mov    %rbx,%rdi
    1f0d:	48 89 c5             	mov    %rax,%rbp
    1f10:	e8 6b 04 00 00       	callq  2380 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1f15:	49 ff cd             	dec    %r13
    1f18:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1f1e:	48 89 ea             	mov    %rbp,%rdx
    1f21:	4c 89 ee             	mov    %r13,%rsi
    1f24:	48 89 df             	mov    %rbx,%rdi
    1f27:	e8 54 04 00 00       	callq  2380 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1f2c:	4d 85 ed             	test   %r13,%r13
    1f2f:	75 e4                	jne    1f15 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x155>
    1f31:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1f38:	49 83 ec 08          	sub    $0x8,%r12
    1f3c:	4c 89 e5             	mov    %r12,%rbp
    1f3f:	48 29 dd             	sub    %rbx,%rbp
    1f42:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1f46:	48 89 ea             	mov    %rbp,%rdx
    1f49:	c4 c1 7b 10 04 24    	vmovsd (%r12),%xmm0
    1f4f:	48 c1 fa 03          	sar    $0x3,%rdx
    1f53:	31 f6                	xor    %esi,%esi
    1f55:	c4 c1 7b 11 0c 24    	vmovsd %xmm1,(%r12)
    1f5b:	48 89 df             	mov    %rbx,%rdi
    1f5e:	e8 1d 04 00 00       	callq  2380 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1f63:	48 83 fd 08          	cmp    $0x8,%rbp
    1f67:	7f cf                	jg     1f38 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x178>
    1f69:	5b                   	pop    %rbx
    1f6a:	5d                   	pop    %rbp
    1f6b:	41 5c                	pop    %r12
    1f6d:	41 5d                	pop    %r13
    1f6f:	41 5e                	pop    %r14
    1f71:	c3                   	retq   
    1f72:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1f78:	c3                   	retq   
    1f79:	90                   	nop
    1f7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>:
    1f80:	f3 0f 1e fa          	endbr64 
    1f84:	41 57                	push   %r15
    1f86:	41 56                	push   %r14
    1f88:	41 55                	push   %r13
    1f8a:	41 54                	push   %r12
    1f8c:	55                   	push   %rbp
    1f8d:	53                   	push   %rbx
    1f8e:	48 83 ec 18          	sub    $0x18,%rsp
    1f92:	48 89 3c 24          	mov    %rdi,(%rsp)
    1f96:	85 d2                	test   %edx,%edx
    1f98:	0f 84 5a 01 00 00    	je     20f8 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1f9e:	48 89 f5             	mov    %rsi,%rbp
    1fa1:	48 85 f6             	test   %rsi,%rsi
    1fa4:	0f 84 4e 01 00 00    	je     20f8 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1faa:	41 89 d4             	mov    %edx,%r12d
    1fad:	49 c1 e4 03          	shl    $0x3,%r12
    1fb1:	4c 89 e0             	mov    %r12,%rax
    1fb4:	41 89 d7             	mov    %edx,%r15d
    1fb7:	48 c1 f8 03          	sar    $0x3,%rax
    1fbb:	ba 3f 00 00 00       	mov    $0x3f,%edx
    1fc0:	f3 48 0f bd c0       	lzcnt  %rax,%rax
    1fc5:	29 c2                	sub    %eax,%edx
    1fc7:	4e 8d 2c 26          	lea    (%rsi,%r12,1),%r13
    1fcb:	48 63 d2             	movslq %edx,%rdx
    1fce:	48 01 d2             	add    %rdx,%rdx
    1fd1:	4c 89 ee             	mov    %r13,%rsi
    1fd4:	48 89 ef             	mov    %rbp,%rdi
    1fd7:	41 89 ce             	mov    %ecx,%r14d
    1fda:	48 8d 5d 08          	lea    0x8(%rbp),%rbx
    1fde:	e8 dd fd ff ff       	callq  1dc0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1fe3:	49 81 fc 80 00 00 00 	cmp    $0x80,%r12
    1fea:	0f 86 30 01 00 00    	jbe    2120 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1a0>
    1ff0:	4c 8d a5 80 00 00 00 	lea    0x80(%rbp),%r12
    1ff7:	eb 3e                	jmp    2037 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    1ff9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2000:	48 39 dd             	cmp    %rbx,%rbp
    2003:	74 24                	je     2029 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xa9>
    2005:	b8 08 00 00 00       	mov    $0x8,%eax
    200a:	48 89 da             	mov    %rbx,%rdx
    200d:	48 29 ea             	sub    %rbp,%rdx
    2010:	48 8d 7c 05 00       	lea    0x0(%rbp,%rax,1),%rdi
    2015:	48 89 ee             	mov    %rbp,%rsi
    2018:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    201e:	e8 bd f0 ff ff       	callq  10e0 <memmove@plt>
    2023:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    2029:	48 83 c3 08          	add    $0x8,%rbx
    202d:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    2032:	49 39 dc             	cmp    %rbx,%r12
    2035:	74 4d                	je     2084 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x104>
    2037:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    203b:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    2040:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2044:	77 ba                	ja     2000 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x80>
    2046:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    204b:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    204f:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2053:	0f 86 67 01 00 00    	jbe    21c0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x240>
    2059:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2060:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    2065:	48 89 c2             	mov    %rax,%rdx
    2068:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    206d:	48 83 e8 08          	sub    $0x8,%rax
    2071:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2075:	77 e9                	ja     2060 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xe0>
    2077:	48 83 c3 08          	add    $0x8,%rbx
    207b:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    207f:	49 39 dc             	cmp    %rbx,%r12
    2082:	75 b3                	jne    2037 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    2084:	4d 39 e5             	cmp    %r12,%r13
    2087:	74 4b                	je     20d4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    2089:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2090:	c4 c1 7b 10 0c 24    	vmovsd (%r12),%xmm1
    2096:	c4 c1 7b 10 44 24 f8 	vmovsd -0x8(%r12),%xmm0
    209d:	49 8d 44 24 f8       	lea    -0x8(%r12),%rax
    20a2:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    20a6:	0f 86 fc 00 00 00    	jbe    21a8 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x228>
    20ac:	0f 1f 40 00          	nopl   0x0(%rax)
    20b0:	c5 fb 11 40 08       	vmovsd %xmm0,0x8(%rax)
    20b5:	48 89 c2             	mov    %rax,%rdx
    20b8:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    20bd:	48 83 e8 08          	sub    $0x8,%rax
    20c1:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    20c5:	77 e9                	ja     20b0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x130>
    20c7:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    20cb:	49 83 c4 08          	add    $0x8,%r12
    20cf:	4d 39 e5             	cmp    %r12,%r13
    20d2:	75 bc                	jne    2090 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x110>
    20d4:	48 8b 3c 24          	mov    (%rsp),%rdi
    20d8:	48 83 c4 18          	add    $0x18,%rsp
    20dc:	5b                   	pop    %rbx
    20dd:	48 89 ee             	mov    %rbp,%rsi
    20e0:	5d                   	pop    %rbp
    20e1:	41 5c                	pop    %r12
    20e3:	41 5d                	pop    %r13
    20e5:	44 89 f1             	mov    %r14d,%ecx
    20e8:	41 5e                	pop    %r14
    20ea:	44 89 fa             	mov    %r15d,%edx
    20ed:	41 5f                	pop    %r15
    20ef:	e9 4c fa ff ff       	jmpq   1b40 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>
    20f4:	0f 1f 40 00          	nopl   0x0(%rax)
    20f8:	48 8b 3d 21 2f 00 00 	mov    0x2f21(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    20ff:	48 83 c4 18          	add    $0x18,%rsp
    2103:	5b                   	pop    %rbx
    2104:	5d                   	pop    %rbp
    2105:	41 5c                	pop    %r12
    2107:	41 5d                	pop    %r13
    2109:	41 5e                	pop    %r14
    210b:	48 8d 15 ce 14 00 00 	lea    0x14ce(%rip),%rdx        # 35e0 <_IO_stdin_used+0x5e0>
    2112:	be 01 00 00 00       	mov    $0x1,%esi
    2117:	31 c0                	xor    %eax,%eax
    2119:	41 5f                	pop    %r15
    211b:	e9 d0 ef ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    2120:	41 bc 08 00 00 00    	mov    $0x8,%r12d
    2126:	49 39 dd             	cmp    %rbx,%r13
    2129:	75 3b                	jne    2166 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1e6>
    212b:	eb a7                	jmp    20d4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    212d:	0f 1f 00             	nopl   (%rax)
    2130:	48 39 dd             	cmp    %rbx,%rbp
    2133:	74 1f                	je     2154 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d4>
    2135:	48 89 da             	mov    %rbx,%rdx
    2138:	48 29 ea             	sub    %rbp,%rdx
    213b:	4a 8d 7c 25 00       	lea    0x0(%rbp,%r12,1),%rdi
    2140:	48 89 ee             	mov    %rbp,%rsi
    2143:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    2149:	e8 92 ef ff ff       	callq  10e0 <memmove@plt>
    214e:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    2154:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    2159:	48 83 c3 08          	add    $0x8,%rbx
    215d:	49 39 dd             	cmp    %rbx,%r13
    2160:	0f 84 6e ff ff ff    	je     20d4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    2166:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    216a:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    216f:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2173:	77 bb                	ja     2130 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1b0>
    2175:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    217a:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    217e:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2182:	76 4c                	jbe    21d0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x250>
    2184:	0f 1f 40 00          	nopl   0x0(%rax)
    2188:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    218d:	48 89 c2             	mov    %rax,%rdx
    2190:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    2195:	48 83 e8 08          	sub    $0x8,%rax
    2199:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    219d:	77 e9                	ja     2188 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x208>
    219f:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    21a3:	eb b4                	jmp    2159 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d9>
    21a5:	0f 1f 00             	nopl   (%rax)
    21a8:	4c 89 e2             	mov    %r12,%rdx
    21ab:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    21af:	49 83 c4 08          	add    $0x8,%r12
    21b3:	e9 17 ff ff ff       	jmpq   20cf <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x14f>
    21b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    21bf:	00 
    21c0:	48 89 da             	mov    %rbx,%rdx
    21c3:	e9 af fe ff ff       	jmpq   2077 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xf7>
    21c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    21cf:	00 
    21d0:	48 89 da             	mov    %rbx,%rdx
    21d3:	eb ca                	jmp    219f <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x21f>
    21d5:	90                   	nop
    21d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    21dd:	00 00 00 

00000000000021e0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE>:
    21e0:	f3 0f 1e fa          	endbr64 
    21e4:	85 d2                	test   %edx,%edx
    21e6:	0f 84 84 00 00 00    	je     2270 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    21ec:	8d 42 ff             	lea    -0x1(%rdx),%eax
    21ef:	83 f8 02             	cmp    $0x2,%eax
    21f2:	0f 86 88 00 00 00    	jbe    2280 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0xa0>
    21f8:	41 89 d0             	mov    %edx,%r8d
    21fb:	41 c1 e8 02          	shr    $0x2,%r8d
    21ff:	41 ff c8             	dec    %r8d
    2202:	49 c1 e0 05          	shl    $0x5,%r8
    2206:	48 89 f0             	mov    %rsi,%rax
    2209:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    220e:	66 90                	xchg   %ax,%ax
    2210:	62 f1 fe 28 7a 00    	vcvtuqq2pd (%rax),%ymm0
    2216:	48 83 c0 20          	add    $0x20,%rax
    221a:	c5 fd 11 40 e0       	vmovupd %ymm0,-0x20(%rax)
    221f:	4c 39 c0             	cmp    %r8,%rax
    2222:	75 ec                	jne    2210 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x30>
    2224:	89 d0                	mov    %edx,%eax
    2226:	83 e0 fc             	and    $0xfffffffc,%eax
    2229:	f6 c2 03             	test   $0x3,%dl
    222c:	74 4a                	je     2278 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x98>
    222e:	c5 f8 77             	vzeroupper 
    2231:	41 89 c0             	mov    %eax,%r8d
    2234:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2238:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    223e:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2243:	44 8d 40 01          	lea    0x1(%rax),%r8d
    2247:	44 39 c2             	cmp    %r8d,%edx
    224a:	76 24                	jbe    2270 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    224c:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2250:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    2256:	83 c0 02             	add    $0x2,%eax
    2259:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    225e:	39 c2                	cmp    %eax,%edx
    2260:	76 0e                	jbe    2270 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    2262:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    2266:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    226c:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    2270:	e9 0b fd ff ff       	jmpq   1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    2275:	0f 1f 00             	nopl   (%rax)
    2278:	c5 f8 77             	vzeroupper 
    227b:	e9 00 fd ff ff       	jmpq   1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    2280:	31 c0                	xor    %eax,%eax
    2282:	eb ad                	jmp    2231 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x51>
    2284:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    228b:	00 00 00 00 
    228f:	90                   	nop

0000000000002290 <_ZN5stats9stats_out9get_statsERSt6vectorIdSaIdEEN6timers10time_unitsE>:
    2290:	f3 0f 1e fa          	endbr64 
    2294:	4c 8b 06             	mov    (%rsi),%r8
    2297:	89 d1                	mov    %edx,%ecx
    2299:	48 8b 56 08          	mov    0x8(%rsi),%rdx
    229d:	4c 89 c6             	mov    %r8,%rsi
    22a0:	4c 29 c2             	sub    %r8,%rdx
    22a3:	48 c1 fa 03          	sar    $0x3,%rdx
    22a7:	e9 d4 fc ff ff       	jmpq   1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    22ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000022b0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE>:
    22b0:	f3 0f 1e fa          	endbr64 
    22b4:	48 89 f0             	mov    %rsi,%rax
    22b7:	89 d1                	mov    %edx,%ecx
    22b9:	48 8b 36             	mov    (%rsi),%rsi
    22bc:	48 8b 50 08          	mov    0x8(%rax),%rdx
    22c0:	48 29 f2             	sub    %rsi,%rdx
    22c3:	48 c1 fa 03          	sar    $0x3,%rdx
    22c7:	85 d2                	test   %edx,%edx
    22c9:	0f 84 82 00 00 00    	je     2351 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    22cf:	8d 42 ff             	lea    -0x1(%rdx),%eax
    22d2:	83 f8 02             	cmp    $0x2,%eax
    22d5:	0f 86 95 00 00 00    	jbe    2370 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xc0>
    22db:	89 d0                	mov    %edx,%eax
    22dd:	c1 e8 02             	shr    $0x2,%eax
    22e0:	48 c1 e0 05          	shl    $0x5,%rax
    22e4:	49 89 f0             	mov    %rsi,%r8
    22e7:	48 01 f0             	add    %rsi,%rax
    22ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    22f0:	62 d1 fe 28 7a 00    	vcvtuqq2pd (%r8),%ymm0
    22f6:	49 83 c0 20          	add    $0x20,%r8
    22fa:	c4 c1 7d 11 40 e0    	vmovupd %ymm0,-0x20(%r8)
    2300:	49 39 c0             	cmp    %rax,%r8
    2303:	75 eb                	jne    22f0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x40>
    2305:	89 d0                	mov    %edx,%eax
    2307:	83 e0 fc             	and    $0xfffffffc,%eax
    230a:	f6 c2 03             	test   $0x3,%dl
    230d:	74 51                	je     2360 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xb0>
    230f:	c5 f8 77             	vzeroupper 
    2312:	41 89 c0             	mov    %eax,%r8d
    2315:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2319:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    231f:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2324:	44 8d 40 01          	lea    0x1(%rax),%r8d
    2328:	44 39 c2             	cmp    %r8d,%edx
    232b:	76 24                	jbe    2351 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    232d:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2331:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    2337:	83 c0 02             	add    $0x2,%eax
    233a:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    233f:	39 c2                	cmp    %eax,%edx
    2341:	76 0e                	jbe    2351 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    2343:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    2347:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    234d:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    2351:	e9 2a fc ff ff       	jmpq   1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    2356:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    235d:	00 00 00 
    2360:	c5 f8 77             	vzeroupper 
    2363:	e9 18 fc ff ff       	jmpq   1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    2368:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    236f:	00 
    2370:	31 c0                	xor    %eax,%eax
    2372:	eb 9e                	jmp    2312 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x62>
    2374:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    237b:	00 00 00 
    237e:	66 90                	xchg   %ax,%ax

0000000000002380 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>:
    2380:	f3 0f 1e fa          	endbr64 
    2384:	48 8d 42 ff          	lea    -0x1(%rdx),%rax
    2388:	49 89 c2             	mov    %rax,%r10
    238b:	49 c1 ea 3f          	shr    $0x3f,%r10
    238f:	49 01 c2             	add    %rax,%r10
    2392:	49 89 d3             	mov    %rdx,%r11
    2395:	49 d1 fa             	sar    %r10
    2398:	41 83 e3 01          	and    $0x1,%r11d
    239c:	4c 39 d6             	cmp    %r10,%rsi
    239f:	0f 8d b3 00 00 00    	jge    2458 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xd8>
    23a5:	49 89 f1             	mov    %rsi,%r9
    23a8:	eb 20                	jmp    23ca <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x4a>
    23aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    23b0:	48 ff c8             	dec    %rax
    23b3:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    23b7:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    23bc:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    23c2:	49 39 c2             	cmp    %rax,%r10
    23c5:	7e 2f                	jle    23f6 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x76>
    23c7:	49 89 c1             	mov    %rax,%r9
    23ca:	49 8d 49 01          	lea    0x1(%r9),%rcx
    23ce:	48 8d 04 09          	lea    (%rcx,%rcx,1),%rax
    23d2:	48 c1 e1 04          	shl    $0x4,%rcx
    23d6:	4c 8d 04 0f          	lea    (%rdi,%rcx,1),%r8
    23da:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    23df:	c5 fb 10 54 0f f8    	vmovsd -0x8(%rdi,%rcx,1),%xmm2
    23e5:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    23e9:	77 c5                	ja     23b0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x30>
    23eb:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    23f1:	49 39 c2             	cmp    %rax,%r10
    23f4:	7f d1                	jg     23c7 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x47>
    23f6:	4d 85 db             	test   %r11,%r11
    23f9:	74 6d                	je     2468 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xe8>
    23fb:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
    23ff:	48 89 d1             	mov    %rdx,%rcx
    2402:	48 c1 e9 3f          	shr    $0x3f,%rcx
    2406:	48 01 d1             	add    %rdx,%rcx
    2409:	48 d1 f9             	sar    %rcx
    240c:	48 39 f0             	cmp    %rsi,%rax
    240f:	7f 2b                	jg     243c <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xbc>
    2411:	eb 3c                	jmp    244f <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    2413:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2418:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
    241c:	48 89 d0             	mov    %rdx,%rax
    241f:	48 c1 e8 3f          	shr    $0x3f,%rax
    2423:	48 01 d0             	add    %rdx,%rax
    2426:	48 d1 f8             	sar    %rax
    2429:	48 89 c2             	mov    %rax,%rdx
    242c:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    2431:	48 89 c8             	mov    %rcx,%rax
    2434:	48 39 ce             	cmp    %rcx,%rsi
    2437:	7d 67                	jge    24a0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x120>
    2439:	48 89 d1             	mov    %rdx,%rcx
    243c:	4c 8d 0c cf          	lea    (%rdi,%rcx,8),%r9
    2440:	c4 c1 7b 10 09       	vmovsd (%r9),%xmm1
    2445:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2449:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    244d:	77 c9                	ja     2418 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x98>
    244f:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2454:	c3                   	retq   
    2455:	0f 1f 00             	nopl   (%rax)
    2458:	4c 8d 04 f7          	lea    (%rdi,%rsi,8),%r8
    245c:	4d 85 db             	test   %r11,%r11
    245f:	75 ee                	jne    244f <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    2461:	48 89 f0             	mov    %rsi,%rax
    2464:	0f 1f 40 00          	nopl   0x0(%rax)
    2468:	48 83 ea 02          	sub    $0x2,%rdx
    246c:	48 89 d1             	mov    %rdx,%rcx
    246f:	48 c1 e9 3f          	shr    $0x3f,%rcx
    2473:	48 01 ca             	add    %rcx,%rdx
    2476:	48 d1 fa             	sar    %rdx
    2479:	48 39 c2             	cmp    %rax,%rdx
    247c:	0f 85 79 ff ff ff    	jne    23fb <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    2482:	48 8d 44 00 02       	lea    0x2(%rax,%rax,1),%rax
    2487:	c5 fb 10 4c c7 f8    	vmovsd -0x8(%rdi,%rax,8),%xmm1
    248d:	48 ff c8             	dec    %rax
    2490:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    2495:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2499:	e9 5d ff ff ff       	jmpq   23fb <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    249e:	66 90                	xchg   %ax,%ax
    24a0:	4d 89 c8             	mov    %r9,%r8
    24a3:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    24a8:	c3                   	retq   
    24a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000024b0 <__libc_csu_init>:
    24b0:	f3 0f 1e fa          	endbr64 
    24b4:	41 57                	push   %r15
    24b6:	4c 8d 3d cb 28 00 00 	lea    0x28cb(%rip),%r15        # 4d88 <__frame_dummy_init_array_entry>
    24bd:	41 56                	push   %r14
    24bf:	49 89 d6             	mov    %rdx,%r14
    24c2:	41 55                	push   %r13
    24c4:	49 89 f5             	mov    %rsi,%r13
    24c7:	41 54                	push   %r12
    24c9:	41 89 fc             	mov    %edi,%r12d
    24cc:	55                   	push   %rbp
    24cd:	48 8d 2d bc 28 00 00 	lea    0x28bc(%rip),%rbp        # 4d90 <__do_global_dtors_aux_fini_array_entry>
    24d4:	53                   	push   %rbx
    24d5:	4c 29 fd             	sub    %r15,%rbp
    24d8:	48 83 ec 08          	sub    $0x8,%rsp
    24dc:	e8 1f eb ff ff       	callq  1000 <_init>
    24e1:	48 c1 fd 03          	sar    $0x3,%rbp
    24e5:	74 1f                	je     2506 <__libc_csu_init+0x56>
    24e7:	31 db                	xor    %ebx,%ebx
    24e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    24f0:	4c 89 f2             	mov    %r14,%rdx
    24f3:	4c 89 ee             	mov    %r13,%rsi
    24f6:	44 89 e7             	mov    %r12d,%edi
    24f9:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    24fd:	48 83 c3 01          	add    $0x1,%rbx
    2501:	48 39 dd             	cmp    %rbx,%rbp
    2504:	75 ea                	jne    24f0 <__libc_csu_init+0x40>
    2506:	48 83 c4 08          	add    $0x8,%rsp
    250a:	5b                   	pop    %rbx
    250b:	5d                   	pop    %rbp
    250c:	41 5c                	pop    %r12
    250e:	41 5d                	pop    %r13
    2510:	41 5e                	pop    %r14
    2512:	41 5f                	pop    %r15
    2514:	c3                   	retq   
    2515:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    251c:	00 00 00 00 

0000000000002520 <__libc_csu_fini>:
    2520:	f3 0f 1e fa          	endbr64 
    2524:	c3                   	retq   

Disassembly of section .fini:

0000000000002528 <_fini>:
    2528:	f3 0f 1e fa          	endbr64 
    252c:	48 83 ec 08          	sub    $0x8,%rsp
    2530:	48 83 c4 08          	add    $0x8,%rsp
    2534:	c3                   	retq   
