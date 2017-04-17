%% savemesh: Saves a 2-D matrix in three views
a = dir('*mat');
for i = 1:size(a,1)
	load(a(i).name);
	name = a(i).name(1:end-4);
	mesh(x);
    colorbar;
	saveas(gcf,strcat(name,'_1.jpg'));
	view(0,0)
	saveas(gcf,strcat(name,'_2.jpg'));
	view(0,-90)
	saveas(gcf,strcat(name,'_3.jpg'));
end