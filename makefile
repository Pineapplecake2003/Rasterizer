gen_golden:
	./script/gen_golden.py | tee result.log
clean:
	rm -rf ./images/
	rm *.log