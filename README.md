To push to the internal registry:
oc patch configs.imageregistry.operator.openshift.io/cluster --patch '{"spec":{"defaultRoute":true}}' --type=merge
oc policy add-role-to-user registry-editor system:serviceaccount:automl:test -n automl
oc adm policy add-role-to-user registry-editor system:serviceaccount:automl:pipeline-runner-dspa -n automl

Here "test" is workbench image and "automl" is the namespace