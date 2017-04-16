%% savemesh: Saves a 2-D matrix in three views
function [] = savemesh(name,mat)
	mesh(mat);
	saveas(gcf,strcat(name,'_1.jpg'));
	view(0,t0)
	saveas(gcf,strcat(name,'_2.jpg'));
	view(0,-90)
	saveas(gcf,strcat(name,'_3.jpg'));
end

