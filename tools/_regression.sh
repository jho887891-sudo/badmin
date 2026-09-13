cd /home/T7/ojh/robot_sim
pass=0; fail=0; total=0; failed=""
for t in $(find tests -name 'test_*.py' | sort); do
  out=$(timeout 240 ./env_isaaclab/bin/python "$t" 2>&1)
  rc=$?
  line=$(echo "$out" | grep -E '^Ran ' | tail -1)
  n=$(echo "$line" | awk '{print $2}')
  [ -z "$n" ] && n=0
  total=$((total+n))
  if [ $rc -eq 0 ]; then pass=$((pass+1)); res=OK; else fail=$((fail+1)); res=FAIL; failed="$failed $t"; fi
  printf '%-58s %-4s rc=%-3s %s\n' "$t" "$res" "$rc" "$line"
done
echo "-----"
echo "suites_passed=$pass suites_failed=$fail tests_total=$total"
echo "failed_suites:$failed"
