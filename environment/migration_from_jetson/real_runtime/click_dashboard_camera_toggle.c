#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XTest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* One bounded click on the CameraRuntime toggle in the known dashboard. */
int main(int argc, char **argv) {
    Display *display;
    Window window;
    XWindowAttributes attrs;
    Window child;
    int root_x = 0, root_y = 0;
    XClassHint class_hint;
    const int click_x = 392;
    const int click_y = 824;

    if (argc != 2) {
        fprintf(stderr, "usage: %s WINDOW_ID\n", argv[0]);
        return 64;
    }
    window = (Window)strtoul(argv[1], NULL, 0);
    display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "cannot open X display\n");
        return 69;
    }
    if (!XGetWindowAttributes(display, window, &attrs) ||
            attrs.width != 1500 || attrs.height != 868) {
        fprintf(stderr, "refusing unexpected dashboard geometry\n");
        XCloseDisplay(display);
        return 78;
    }
    memset(&class_hint, 0, sizeof(class_hint));
    if (!XGetClassHint(display, window, &class_hint) ||
            !class_hint.res_name ||
            strcmp(class_hint.res_name, "elfin_vision_dashboard.py") != 0) {
        fprintf(stderr, "refusing unexpected window class\n");
        if (class_hint.res_name) XFree(class_hint.res_name);
        if (class_hint.res_class) XFree(class_hint.res_class);
        XCloseDisplay(display);
        return 78;
    }
    XFree(class_hint.res_name);
    if (class_hint.res_class) XFree(class_hint.res_class);
    if (!XTranslateCoordinates(display, window, attrs.root, click_x, click_y,
            &root_x, &root_y, &child)) {
        fprintf(stderr, "cannot translate dashboard coordinates\n");
        XCloseDisplay(display);
        return 69;
    }
    XRaiseWindow(display, window);
    XSync(display, False);
    usleep(250000);
    XTestFakeMotionEvent(display, DefaultScreen(display), root_x, root_y, 0);
    XSync(display, False);
    usleep(100000);
    XTestFakeButtonEvent(display, 1, True, 0);
    XSync(display, False);
    usleep(80000);
    XTestFakeButtonEvent(display, 1, False, 0);
    XSync(display, False);
    usleep(250000);
    printf("dashboard_camera_toggle_clicked window=0x%lx x=%d y=%d root_x=%d root_y=%d\n",
           (unsigned long)window, click_x, click_y, root_x, root_y);
    XCloseDisplay(display);
    return 0;
}
