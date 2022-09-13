При CBC солвера с помощью библиотеки Pyomo в Python, часто при завершении по time limit CBC не успевает корректно сохранить текущее решение.
Проблема в том, что в pyomo есть два ограничения по времени. Одно передается солверу, а другое контролируется кодом вызывающим солвер.
Если зазор между ними небольшой, то солвер не успевает корректно завершиться.
Поэтому в файле C:\Anaconda3\Lib\site-packages\pyomo\solvers\plugins\solvers\CBCplugin.py
следует перед строками
            if self._timelimit is not None and self._timelimit > 0.0:
                cmd.extend(['-sec', str(self._timelimit - 1 )])
                cmd.extend(['-timeMode', "elapsed"])
которые, почему-то не работают, написать нечто подобное:
            if self.options['seconds'] is not None and self.options['seconds'] > 0.0:
                self.options['seconds'] -= 1
                print('Time limit corrected (by amax):', self.options['seconds'])
                cmd.extend(['-timeMode', "elapsed"])
И тогда времени на корректное завершение хватает.


Цельный фрагмент исправленного кода
            print ('AMAX1')
            if self._timelimit is not None and self._timelimit > 0.0:
#                cmd.extend(['-sec', str(self._timelimit)])
                cmd.extend(['-sec', str(self._timelimit - 1 )])
                cmd.extend(['-timeMode', "elapsed"])
            if "debug" in self.options:
                cmd.extend(["-log","5"])
            for key, val in _check_and_escape_options(self.options):
                if key == 'solver':
                    continue
                cmd.append(key+"="+val)
            os.environ['cbc_options']="printingOptions=all"
            #cmd.extend(["-printingOptions=all",
                        #"-stat"])
        else:
            if self.options['seconds'] is not None and self.options['seconds'] > 0.0:
                self.options['seconds'] -= 1
                print('Time limit corrected (by amax):', self.options['seconds'])
                cmd.extend(['-timeMode', "elapsed"])
            if self._timelimit is not None and self._timelimit > 0.0:
#                cmd.extend(['-sec', str(self._timelimit)])
                #print ('AMAX3 tlim =', self._timelimit - 1 )
                cmd.extend(['-sec', str(self._timelimit - 1 )])
                cmd.extend(['-timeMode', "elapsed"])
